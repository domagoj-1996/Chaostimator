#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import pandas as pd
import numpy as np
import os
import math
import re
import time

# -----------------------
# Config
# -----------------------
ROLL_SIZES = [4, 5, 6]                 # total affixes
ROLL_WEIGHTS = [1.0, 3.0, 8.0]        # relative weights for roll sizes 4,5,6 (1:3:8)
ROLL_PROBS = np.array(ROLL_WEIGHTS) / sum(ROLL_WEIGHTS)
MAX_PREFIX = 3
MAX_SUFFIX = 3

MC_SAMPLES_DEFAULT = 2000  # fast-mode default

# -----------------------
# Helpers
# -----------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        key = c.replace(" ", "_").replace("-", "_")
        rename_map[c] = key
    df.rename(columns=rename_map, inplace=True)
    return df

def parse_exclusive_cell(v):
    if pd.isna(v) or str(v).strip() == "":
        return []
    s = str(v).strip()
    parts = re.split(r"[,\s;]+", s)
    return [p.strip() for p in parts if p.strip()]

def tier_is_17(t):
    if t is None:
        return False
    s = str(t).strip().lower()
    if s == "":
        return False
    s = s.lstrip('t')  # remove leading 't' if present
    return s == "17" or s == "17.0"

# -----------------------
# Accurate estimator (deterministic enumeration on reduced pools)
# -----------------------
class AccurateEstimator:
    def __init__(self, df: pd.DataFrame, progress_callback=None, stop_event=None):
        """
        df: full mods DataFrame (normalized)
        """
        self.orig_df = df.copy().reset_index(drop=True)
        self.progress_callback = progress_callback
        self.stop_event = stop_event
        self.cols = ["quantity","rarity","pack_size","currency","scarabs","maps"]

        # normalize type
        def map_type(v):
            try:
                vn = float(v)
                if int(vn) == 1:
                    return "prefix"
                if int(vn) == 2:
                    return "suffix"
            except Exception:
                pass
            s = str(v).strip().lower()
            if "prefix" in s:
                return "prefix"
            if "suffix" in s:
                return "suffix"
            return "prefix"
        self.orig_df["type"] = self.orig_df["type"].apply(map_type)
        for c in self.cols:
            if c not in self.orig_df.columns:
                self.orig_df[c] = 0.0
            self.orig_df[c] = pd.to_numeric(self.orig_df[c], errors="coerce").fillna(0.0)
        self.orig_df["weight"] = pd.to_numeric(self.orig_df.get("weight",0.0), errors="coerce").fillna(0.0)
        self.orig_df["mod_group"] = self.orig_df.get("mod_group","").astype(str).fillna("").apply(lambda x:x.strip())
        self.orig_df["exclusive_with"] = self.orig_df.get("exclusive_with","").astype(str).fillna("").apply(lambda x:x.strip())
        self.orig_df["tier"] = self.orig_df.get("tier","").astype(str).fillna("").apply(lambda x:x.strip())
        self.orig_df["excluded"] = self.orig_df.get("excluded", False)

    def _build_records(self, df):
        pref = []
        suf = []
        for i, r in df.iterrows():
            if bool(r.get("excluded", False)):
                continue
            rec = {
                "idx": int(i),
                "mod": str(r["mod"]),
                "group": str(r.get("mod_group","")).strip(),
                "excl": set(parse_exclusive_cell(r.get("exclusive_with",""))),
                "weight": float(r.get("weight",0.0)),
                "tier": str(r.get("tier","")).strip(),
                "stats": np.array([float(r.get(c,0.0)) for c in self.cols], dtype=float)
            }
            if r["type"] == "prefix":
                pref.append(rec)
            else:
                suf.append(rec)
        return pref, suf

    def _apply_conflicts(self, selected, pref_pool, suf_pool):
        """
        Remove mods that conflict with 'selected' (selected is a single record)
        Conflicts rules (bidirectional):
          - same mod_group string => remove
          - if selected.excl contains other.idx (string) or other.group => remove
          - if other.excl contains selected.idx or selected.group => remove
        """
        if selected is None:
            return pref_pool, suf_pool
        sg = selected.get("group","")
        sex = selected.get("excl", set())

        def allowed(rec):
            cg = rec.get("group","")
            cex = rec.get("excl", set())
            # same mod_group
            if cg != "" and sg != "" and cg == sg:
                return False
            # selected explicitly excludes candidate id or group
            if str(rec["idx"]) in sex:
                return False
            if cg != "" and cg in sex:
                return False
            # candidate excludes selected id or selected.group
            if str(selected["idx"]) in cex:
                return False
            if sg != "" and sg in cex:
                return False
            return True

        new_pref = [r for r in pref_pool if allowed(r)]
        new_suf = [r for r in suf_pool if allowed(r)]
        return new_pref, new_suf

    def compute_probability(self, thresholds: dict):
        """
        Compute exact probability by enumerating only relevant mods.
        thresholds: dict mapping stat->value
        """
        thr = np.array([float(thresholds.get(c, 0.0)) for c in self.cols], dtype=float)
        # Which stats are active (user set > 0)
        active_mask = thr > 0

        # Build full records (no exclusions applied yet) - we need for guaranteed T17 picks
        prefix_records, suffix_records = self._build_records(self.orig_df)

        # Need at least 1 T17 in both prefix & suffix
        t17_pref = [r for r in prefix_records if tier_is_17(r["tier"])]
        t17_suf = [r for r in suffix_records if tier_is_17(r["tier"])]
        if not t17_pref or not t17_suf:
            return 0.0

        # For performance: create reduced pools for extra picks - include mods that could affect any active stat
        def relevant(rec):
            if not any(active_mask):
                # no thresholds: everything is relevant (rare case)
                return True
            # include if rec has any stat for active dimensions
            return rec["stats"][active_mask].sum() > 0.0

        # Note: for extra picks we must consider both T16 and T17; but we reduce to mods relevant to thresholds
        reduced_prefix_all = [r for r in prefix_records if relevant(r)]
        reduced_suffix_all = [r for r in suffix_records if relevant(r)]

        # Precompute total T17 weights (for guaranteed picks)
        total_t17_pref_w = sum(max(0.0, r["weight"]) for r in t17_pref)
        total_t17_suf_w = sum(max(0.0, r["weight"]) for r in t17_suf)
        if total_t17_pref_w <= 0 or total_t17_suf_w <= 0:
            return 0.0

        total_success_prob = 0.0

        # For each roll size (k), compute splits and their contribution
        for j, k in enumerate(ROLL_SIZES):
            roll_prob = float(ROLL_PROBS[j])
            # valid splits:
            max_p = min(MAX_PREFIX, k)
            max_s = min(MAX_SUFFIX, k)
            splits = [(p, k-p) for p in range(0, k+1) if p <= max_p and (k-p) <= max_s]
            split_prob = 1.0 / len(splits)

            # iterate guaranteed T17 prefix choices
            for pref_choice in t17_pref:
                w_pref_choice = max(0.0, pref_choice["weight"])
                prob_pref_choice = w_pref_choice / total_t17_pref_w if total_t17_pref_w > 0 else 0.0
                if prob_pref_choice == 0.0:
                    continue

                # pools after selecting this T17 prefix (apply conflicts to original pools)
                pref_after_p17, suf_after_p17 = self._apply_conflicts(pref_choice, prefix_records, suffix_records)
                # remove the chosen from pref_after_p17
                pref_after_p17 = [r for r in pref_after_p17 if r["idx"] != pref_choice["idx"]]

                for suf_choice in t17_suf:
                    w_suf_choice = max(0.0, suf_choice["weight"])
                    prob_suf_choice = w_suf_choice / total_t17_suf_w if total_t17_suf_w > 0 else 0.0
                    if prob_suf_choice == 0.0:
                        continue

                    # pools after both mandatory picks
                    pref_pool_init, suf_pool_init = self._apply_conflicts(suf_choice, pref_after_p17, suf_after_p17)
                    suf_pool_init = [r for r in suf_pool_init if r["idx"] != suf_choice["idx"]]

                    # reduce pools to only relevant mods (for speed)
                    pref_pool_reduced = [r for r in pref_pool_init if relevant(r)]
                    suf_pool_reduced = [r for r in suf_pool_init if relevant(r)]

                    # iterate splits
                    for p_total, s_total in splits:
                        rem_p = p_total - 1
                        rem_s = s_total - 1
                        if rem_p < 0 or rem_s < 0:
                            continue
                        if rem_p > len(pref_pool_reduced) or rem_s > len(suf_pool_reduced):
                            # impossible to pick required distinct mods from reduced pool -> skip
                            continue

                        # quick upper bound pruning: compute maximum possible of active stats
                        # guaranteed (from T17 mandatory picks)
                        guaranteed = pref_choice["stats"] + suf_choice["stats"]
                        # top rem_p from pref_pool_reduced
                        def top_sum(pool, picks):
                            if picks <= 0 or not pool:
                                return np.zeros(len(self.cols), dtype=float)
                            scores = sorted(pool, key=lambda r: r["stats"].sum(), reverse=True)
                            top = scores[:picks]
                            res = np.zeros(len(self.cols), dtype=float)
                            for r in top:
                                res += r["stats"]
                            return res
                        max_pref_stats = top_sum(pref_pool_reduced, rem_p)
                        max_suf_stats = top_sum(suf_pool_reduced, rem_s)
                        max_possible = guaranteed + max_pref_stats + max_suf_stats

                        # if for any active stat threshold greater than max_possible, impossible
                        impossible = False
                        for idx in range(len(thr)):
                            if thr[idx] > 0 and max_possible[idx] < thr[idx]:
                                impossible = True
                                break
                        if impossible:
                            continue

                        # Now enumerate prefix picks exactly (recursive weighted without replacement + dynamic exclusions)
                        # We'll collect prefix outcomes: (pref_sum, pref_prob, suf_pool_after_prefix)
                        prefix_outcomes = []

                        def recurse_prefix(av_pref, av_suf, picks_left, cur_sum, cur_prob):
                            # cancellation
                            if self.stop_event and getattr(self.stop_event, "is_set", lambda: False)():
                                return
                            if picks_left == 0:
                                prefix_outcomes.append((cur_sum.copy(), cur_prob, [r for r in av_suf]))
                                return
                            total_w = sum(max(0.0, r["weight"]) for r in av_pref)
                            if total_w <= 0.0:
                                prefix_outcomes.append((cur_sum.copy(), cur_prob, [r for r in av_suf]))
                                return
                            for i in range(len(av_pref)):
                                r = av_pref[i]
                                w = max(0.0, r["weight"])
                                prob_pick = w / total_w if total_w > 0 else 0.0
                                # build new pools and apply conflicts
                                new_pref = [x for xi,x in enumerate(av_pref) if xi != i]
                                new_pref_filtered, new_suf_filtered = self._apply_conflicts(r, new_pref, av_suf)
                                new_sum = cur_sum + r["stats"]
                                recurse_prefix(new_pref_filtered, new_suf_filtered, picks_left - 1, new_sum, cur_prob * prob_pick)

                        recurse_prefix(pref_pool_reduced.copy(), suf_pool_reduced.copy(), rem_p, np.zeros(len(self.cols), dtype=float), 1.0)

                        # For each prefix outcome, enumerate suffix outcomes similarly
                        for pref_sum, pref_prob, suf_pool_after_pref in prefix_outcomes:
                            # quick prune: compute max possible suffix remaining
                            max_suf_remain = top_sum(suf_pool_after_pref, rem_s)
                            curr_possible = pref_choice["stats"] + suf_choice["stats"] + pref_sum + max_suf_remain
                            skip = False
                            for idx in range(len(thr)):
                                if thr[idx] > 0 and curr_possible[idx] < thr[idx]:
                                    skip = True
                                    break
                            if skip:
                                continue

                            # enumerate suffix sequences
                            def recurse_suffix(av_suf, picks_left, cur_sum, cur_prob):
                                if picks_left == 0:
                                    return [(cur_sum.copy(), cur_prob)]
                                total_w = sum(max(0.0, r["weight"]) for r in av_suf)
                                if total_w <= 0.0:
                                    return [(cur_sum.copy(), cur_prob)]
                                res = []
                                for i in range(len(av_suf)):
                                    r = av_suf[i]
                                    w = max(0.0, r["weight"])
                                    prob_pick = w / total_w if total_w > 0 else 0.0
                                    new_suf = [x for xi,x in enumerate(av_suf) if xi != i]
                                    # remove conflicts caused by this pick (only suffixes & prefixes affected)
                                    _, new_suf_filtered = self._apply_conflicts(r, [], new_suf)
                                    new_sum = cur_sum + r["stats"]
                                    res.extend(recurse_suffix(new_suf_filtered, picks_left - 1, new_sum, cur_prob * prob_pick))
                                return res

                            suffix_outcomes = recurse_suffix(suf_pool_after_pref.copy(), rem_s, np.zeros(len(self.cols), dtype=float), 1.0)

                            # evaluate joint outcomes
                            for suf_sum, suf_prob in suffix_outcomes:
                                total_stats = pref_choice["stats"] + suf_choice["stats"] + pref_sum + suf_sum
                                joint_prob = roll_prob * split_prob * prob_pref_choice * prob_suf_choice * pref_prob * suf_prob
                                # check all thresholds
                                ok = True
                                for idx in range(len(thr)):
                                    if thr[idx] > 0 and total_stats[idx] < thr[idx]:
                                        ok = False
                                        break
                                if ok:
                                    total_success_prob += joint_prob

            # optional progress callback per roll size
            if self.progress_callback:
                self.progress_callback(None)

        return total_success_prob

# -----------------------
# Fast estimator (optimized Monte-Carlo on reduced pools)
# -----------------------
class FastEstimator:
    def __init__(self, df: pd.DataFrame, samples=MC_SAMPLES_DEFAULT, rng_seed=None):
        self.orig_df = df.copy().reset_index(drop=True)
        self.samples = int(samples)
        self.rng = np.random.default_rng(rng_seed if rng_seed is not None else None)
        self.cols = ["quantity","rarity","pack_size","currency","scarabs","maps"]
        # normalize type
        def map_type(v):
            try:
                vn = float(v)
                if int(vn) == 1:
                    return "prefix"
                if int(vn) == 2:
                    return "suffix"
            except Exception:
                pass
            s = str(v).strip().lower()
            if "prefix" in s:
                return "prefix"
            if "suffix" in s:
                return "suffix"
            return "prefix"
        self.orig_df["type"] = self.orig_df["type"].apply(map_type)
        for c in self.cols:
            if c not in self.orig_df.columns:
                self.orig_df[c] = 0.0
            self.orig_df[c] = pd.to_numeric(self.orig_df[c], errors="coerce").fillna(0.0)
        self.orig_df["weight"] = pd.to_numeric(self.orig_df.get("weight",0.0), errors="coerce").fillna(0.0)
        self.orig_df["mod_group"] = self.orig_df.get("mod_group","").astype(str).fillna("").apply(lambda x:x.strip())
        self.orig_df["exclusive_with"] = self.orig_df.get("exclusive_with","").astype(str).fillna("").apply(lambda x:x.strip())
        self.orig_df["tier"] = self.orig_df.get("tier","").astype(str).fillna("").apply(lambda x:x.strip())
        self.orig_df["excluded"] = self.orig_df.get("excluded", False)

    def _build_compact_pools(self, thresholds):
        """
        Build compact prefix and suffix lists containing only mods that can affect thresholds,
        but keep all T17s for guaranteed picks (even if they lack the stat) because their weight matters.
        """
        thr = np.array([float(thresholds.get(c,0.0)) for c in self.cols], dtype=float)
        active_mask = thr > 0

        pref = []
        suf = []
        for i, r in self.orig_df.iterrows():
            if bool(r.get("excluded", False)):
                continue
            excl_set = set(parse_exclusive_cell(r.get("exclusive_with","")))
            rec = {
                "idx": int(i),
                "mod": str(r["mod"]),
                "group": str(r.get("mod_group","")).strip(),
                "excl_set": excl_set,
                "weight": float(r.get("weight",0.0)),
                "tier": str(r.get("tier","")).strip(),
                "stats": np.array([float(r.get(c,0.0)) for c in self.cols], dtype=float)
            }
            # if a mod is T17, keep it always (guaranteed pick pool)
            if r["type"] == "prefix":
                pref.append(rec)
            else:
                suf.append(rec)

        return pref, suf

    def _apply_conflicts(self, selected, pref_pool, suf_pool):
        if selected is None:
            return pref_pool, suf_pool
        sg = selected.get("group","")
        sex = set(selected.get("excl_set", set()))
        def allowed(rec):
            cg = rec.get("group","")
            cex = rec.get("excl_set", set())
            if cg != "" and sg != "" and cg == sg:
                return False
            if str(rec["idx"]) in sex:
                return False
            if cg != "" and cg in sex:
                return False
            if str(selected["idx"]) in cex:
                return False
            if sg != "" and sg in cex:
                return False
            return True
        new_pref = [r for r in pref_pool if allowed(r)]
        new_suf = [r for r in suf_pool if allowed(r)]
        return new_pref, new_suf

    def _weighted_choice(self, pool):
        if not pool:
            return None
        weights = np.array([max(0.0, r["weight"]) for r in pool], dtype=float)
        total = weights.sum()
        if total <= 0:
            return pool[int(self.rng.integers(len(pool)))]
        probs = weights / total
        idx = int(self.rng.choice(len(pool), p=probs))
        return pool[idx]

    def sample_once(self, thresholds):
        pref_pool, suf_pool = self._build_compact_pools(thresholds)

        # mandatory T17 prefix
        t17_pref = [r for r in pref_pool if tier_is_17(r["tier"])]
        t17_suf = [r for r in suf_pool if tier_is_17(r["tier"])]
        if not t17_pref or not t17_suf:
            return False

        # pick T17 prefix weighted
        weights = np.array([max(0.0, r["weight"]) for r in t17_pref], dtype=float)
        if weights.sum() <= 0:
            chosen_pref = t17_pref[int(self.rng.integers(len(t17_pref)))]
        else:
            chosen_pref = t17_pref[int(self.rng.choice(len(t17_pref), p=weights/weights.sum()))]

        pref_pool, suf_pool = self._apply_conflicts(chosen_pref, pref_pool, suf_pool)
        pref_pool = [r for r in pref_pool if r["idx"] != chosen_pref["idx"]]

        # pick T17 suffix weighted (ensure pick from remaining suf_pool)
        suf_pool_ids = set(r["idx"] for r in suf_pool)
        t17_suf_remaining = [r for r in t17_suf if r["idx"] in suf_pool_ids]
        if not t17_suf_remaining:
            return False
        weights2 = np.array([max(0.0, r["weight"]) for r in t17_suf_remaining], dtype=float)
        if weights2.sum() <= 0:
            chosen_suf = t17_suf_remaining[int(self.rng.integers(len(t17_suf_remaining)))]
        else:
            chosen_suf = t17_suf_remaining[int(self.rng.choice(len(t17_suf_remaining), p=weights2/weights2.sum()))]

        pref_pool, suf_pool = self._apply_conflicts(chosen_suf, pref_pool, suf_pool)
        suf_pool = [r for r in suf_pool if r["idx"] != chosen_suf["idx"]]

        # choose total roll size k
        k = int(self.rng.choice(ROLL_SIZES, p=ROLL_PROBS))
        max_p = min(MAX_PREFIX, k)
        max_s = min(MAX_SUFFIX, k)
        splits = [(p, k-p) for p in range(0, k+1) if p <= max_p and (k-p) <= max_s]
        p_total, s_total = splits[int(self.rng.integers(len(splits)))]
        rem_p = p_total - 1
        rem_s = s_total - 1

        # quick impossibility check
        if rem_p > len(pref_pool) or rem_s > len(suf_pool):
            return False

        # sequential picks (optimized weighted sampling)
        pref_sum = chosen_pref["stats"].copy()
        suf_sum = chosen_suf["stats"].copy()

        for _ in range(rem_p):
            pick = self._weighted_choice(pref_pool)
            if pick is None:
                return False
            pref_sum += pick["stats"]
            pref_pool, suf_pool = self._apply_conflicts(pick, pref_pool, suf_pool)
            pref_pool = [r for r in pref_pool if r["idx"] != pick["idx"]]

        for _ in range(rem_s):
            pick = self._weighted_choice(suf_pool)
            if pick is None:
                return False
            suf_sum += pick["stats"]
            pref_pool, suf_pool = self._apply_conflicts(pick, pref_pool, suf_pool)
            suf_pool = [r for r in suf_pool if r["idx"] != pick["idx"]]

        total_stats = pref_sum + suf_sum
        thr = np.array([float(thresholds.get(c,0.0)) for c in self.cols], dtype=float)
        return (total_stats >= thr).all()

    def compute_probability(self, thresholds, samples=None, progress_callback=None, stop_event=None):
        if samples is None:
            samples = self.samples
        samples = int(samples)
        # quick check for T17 existence in pools
        pref_pool, suf_pool = self._build_compact_pools(thresholds)
        if not any(tier_is_17(r["tier"]) for r in pref_pool) or not any(tier_is_17(r["tier"]) for r in suf_pool):
            return 0.0
        hits = 0
        for i in range(samples):
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                return None
            if self.sample_once(thresholds):
                hits += 1
            if progress_callback and (i % 500 == 0):
                progress_callback(i / samples)
        return float(hits) / float(samples) if samples > 0 else 0.0

# -----------------------
# GUI App (keeps layout you used)
# -----------------------
class ChaosOrbApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chaostimator (Accurate / Fast)")
        self.geometry("1280x720")

        self.mods_df = None
        self.mode_var = tk.StringVar(value="Accurate")  # Accurate or Fast
        self.approx_samples_var = tk.StringVar(value=str(MC_SAMPLES_DEFAULT))

        self._build_ui()
        self.auto_load()

    def auto_load(self):
        folder = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(folder, "data.xlsx")
        if os.path.exists(path):
            self.load_excel(path)
        else:
            self.status_var.set("Ready. Place data.xlsx in app folder or click Load.")

    def load_excel(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(title="Open mods Excel", filetypes=[("Excel files","*.xlsx *.xls")])
        if not path:
            return
        try:
            df = pd.read_excel(path)
        except Exception as e:
            messagebox.showerror("Error loading Excel", str(e))
            return

        df = normalize_columns(df)
        expected = ["mod","type","quantity","rarity","pack_size","currency","scarabs","maps","weight","mod_group","exclusive_with","tier"]
        for col in expected:
            if col not in df.columns:
                if col in ("mod_group","exclusive_with","tier"):
                    df[col] = ""
                elif col == "weight":
                    df[col] = 1.0
                else:
                    df[col] = 0.0

        df["mod"] = df["mod"].astype(str)
        df["type"] = df["type"].astype(str)
        for c in ["quantity","rarity","pack_size","currency","scarabs","maps"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        df["mod_group"] = df.get("mod_group","").astype(str).fillna("").apply(lambda x:x.strip())
        df["exclusive_with"] = df.get("exclusive_with","").astype(str).fillna("").apply(lambda x:x.strip())
        df["tier"] = df.get("tier","").astype(str).fillna("").apply(lambda x:x.strip())
        df["excluded"] = df.get("excluded", False)

        self.mods_df = df.reset_index(drop=True)
        self.status_var.set(f"Loaded {len(self.mods_df)} mods from {os.path.basename(path)}")
        self.populate_tree()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=8, pady=8)

        self.load_btn = ttk.Button(top, text="Load Mods Excel", command=self.load_excel)
        self.load_btn.pack(side="left")
        self.save_btn = ttk.Button(top, text="Save Filtered Mods", command=self.save_filtered_mods)
        self.save_btn.pack(side="right")

        middle = ttk.Panedwindow(self, orient="horizontal")
        middle.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        # left: tree + filter
        left_frame = ttk.Frame(middle, width=820)
        middle.add(left_frame, weight=3)

        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(side="top", fill="x", pady=(0,6))
        ttk.Label(filter_frame, text="Filter mods by name:").pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self.apply_filter())
        self.filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=36)
        self.filter_entry.pack(side="left", padx=(6,0))
        self.clear_exclusions_btn = ttk.Button(filter_frame, text="Clear Exclusions", command=self.clear_exclusions)
        self.clear_exclusions_btn.pack(side="right")

        cols = ("Exclude","Mod","quantity","rarity","pack_size","currency","scarabs","maps")
        self.tree = ttk.Treeview(left_frame, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c, command=lambda _c=c: self.sort_tree(_c, False))
            if c == "Mod":
                self.tree.column(c, width=420, anchor="w")
            else:
                self.tree.column(c, width=90, anchor="center")
        vsb = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self._on_tree_double_click)

        # right: thresholds + controls + results
        right_frame = ttk.Frame(middle, width=380)
        middle.add(right_frame, weight=1)

        th_group = ttk.LabelFrame(right_frame, text="Minimum thresholds (sum of affixes on roll)")
        th_group.pack(fill="x", padx=6, pady=6)
        self.threshold_vars = {}
        for name in ["currency","scarabs","maps","quantity","rarity","pack_size"]:
            row = ttk.Frame(th_group)
            row.pack(fill="x", padx=6, pady=4)
            ttk.Label(row, text=f"{name}:").pack(side="left")
            v = tk.StringVar(value="0")
            ttk.Entry(row, textvariable=v, width=14).pack(side="right")
            self.threshold_vars[name] = v

        mode_frame = ttk.Frame(right_frame)
        mode_frame.pack(fill="x", padx=6, pady=6)
        ttk.Label(mode_frame, text="Mode:").pack(side="left")
        mode_menu = ttk.Combobox(mode_frame, textvariable=self.mode_var, values=["Accurate","Fast"], state="readonly", width=10)
        mode_menu.pack(side="left", padx=(6,6))
        ttk.Label(mode_frame, text="Fast samples:").pack(side="left")
        ttk.Entry(mode_frame, textvariable=self.approx_samples_var, width=12).pack(side="left", padx=(6,0))

        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill="x", padx=6, pady=6)
        self.run_btn = ttk.Button(btn_frame, text="Calculate", command=self.run_calc)
        self.run_btn.pack(side="left", padx=(0,6))
        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.cancel_calc, state="disabled")
        self.cancel_btn.pack(side="left")

        self.progress = ttk.Progressbar(right_frame, orient="horizontal", mode="indeterminate")
        self.progress.pack(fill="x", padx=6, pady=(6,0))

        res_group = ttk.LabelFrame(right_frame, text="Results")
        res_group.pack(fill="both", expand=True, padx=6, pady=6)
        bold_font = tkfont.Font(size=11, weight="bold")
        self.result_label = tk.Label(res_group, text="No results yet.", wraplength=360, justify="left", font=bold_font, anchor="nw")
        self.result_label.pack(fill="both", expand=True, padx=8, pady=8)

        self.status_var = tk.StringVar(value="Ready.")
        status_label = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_label.pack(side="bottom", fill="x", padx=8, pady=6)

    def populate_tree(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        if self.mods_df is None:
            return
        for i, r in self.mods_df.iterrows():
            mod_display = str(r["mod"]).splitlines()[0]
            excl = "X" if bool(r.get("excluded", False)) else ""
            vals = (excl, mod_display, r["quantity"], r["rarity"], r["pack_size"], r["currency"], r["scarabs"], r["maps"])
            self.tree.insert("", "end", iid=str(i), values=vals)

    def _on_tree_double_click(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        idx = int(item)
        self.mods_df.at[idx, "excluded"] = not bool(self.mods_df.at[idx, "excluded"])
        vals = list(self.tree.item(item, "values"))
        vals[0] = "X" if self.mods_df.at[idx, "excluded"] else ""
        self.tree.item(item, values=vals)

    def apply_filter(self):
        txt = self.filter_var.get().strip().lower()
        if txt == "":
            self.populate_tree()
            return
        # show only matching
        for iid in list(self.tree.get_children()):
            self.tree.delete(iid)
        for i, r in self.mods_df.iterrows():
            name = str(r["mod"]).splitlines()[0].lower()
            if txt in name:
                excl = "X" if bool(r.get("excluded", False)) else ""
                vals = (excl, str(r["mod"]).splitlines()[0], r["quantity"], r["rarity"], r["pack_size"], r["currency"], r["scarabs"], r["maps"])
                self.tree.insert("", "end", iid=str(i), values=vals)

    def sort_tree(self, col, descending=False):
        data = []
        for child in self.tree.get_children():
            val = self.tree.set(child, col)
            data.append((val, child))
        def try_float(x):
            try:
                return float(x)
            except Exception:
                return None
        numeric = all(try_float(v) is not None for v,_ in data)
        if numeric:
            data.sort(key=lambda t: float(t[0]), reverse=descending)
        else:
            data.sort(key=lambda t: t[0].lower(), reverse=descending)
        for index, (_, iid) in enumerate(data):
            self.tree.move(iid, "", index)
        self.tree.heading(col, command=lambda: self.sort_tree(col, not descending))

    def clear_exclusions(self):
        if self.mods_df is None:
            return
        self.mods_df["excluded"] = False
        self.populate_tree()

    def save_filtered_mods(self):
        if self.mods_df is None:
            return
        df = self.mods_df[~self.mods_df["excluded"]].reset_index(drop=True)
        if df.empty:
            messagebox.showinfo("No mods", "No mods to save (all excluded).")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files","*.xlsx")])
        if not path:
            return
        try:
            df.to_excel(path, index=False)
            messagebox.showinfo("Saved", f"Saved {len(df)} mods to {path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def run_calc(self):
        if self.mods_df is None:
            messagebox.showwarning("No data", "Load data.xlsx first.")
            return
        # read thresholds
        try:
            thresholds = {k: float(v.get()) for k, v in self.threshold_vars.items()}
        except Exception:
            messagebox.showwarning("Invalid thresholds", "Enter numeric thresholds.")
            return

        mode = self.mode_var.get()
        self.cancel_flag = False
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress.start(80)
        self.status_var.set("Computing...")

        # run after to keep UI responsive
        self.after(50, lambda: self._compute_background(mode, thresholds))

    def cancel_calc(self):
        self.cancel_flag = True
        self.status_var.set("Cancelling...")

    def _compute_background(self, mode, thresholds):
        start = time.time()
        if mode == "Accurate":
            estimator = AccurateEstimator(self.mods_df, progress_callback=None, stop_event=self)
            prob = estimator.compute_probability(thresholds)
        else:
            # Fast (Monte-Carlo over reduced pools)
            try:
                samples = int(self.approx_samples_var.get())
            except Exception:
                samples = MC_SAMPLES_DEFAULT
            estimator = FastEstimator(self.mods_df, samples)
            prob = estimator.compute_probability(thresholds, samples, progress_callback=None, stop_event=self)

        self.progress.stop()
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        if prob is None:
            self.status_var.set("Cancelled.")
            self.result_label.config(text="Calculation cancelled.")
            return
        prob_pct = prob * 100.0
        avg = math.ceil(1.0 / prob) if prob > 0 else "∞"
        self.status_var.set(f"Done in {time.time()-start:.2f}s")
        self.result_label.config(text=f"Estimated probability per Chaos Orb roll: {prob_pct:.3f}%\nAverage Chaos Orbs: {avg}")

    # stop_event API
    def is_set(self):
        return getattr(self, "cancel_flag", False)

# -----------------------
# Entry point
# -----------------------
def main():
    app = ChaosOrbApp()
    app.mainloop()

if __name__ == "__main__":
    main()
