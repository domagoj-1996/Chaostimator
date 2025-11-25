#!/usr/bin/env python3
"""
Chaostimator.py
Deterministic Chaos Orb Outcome Estimator (Tkinter)
- Weighted mods
- Prefix / Suffix distinction (type column: 1 = prefix, 2 = suffix)
- Random prefix/suffix split per roll (4-6 affixes, max 3 each)
- Deterministic Monte Carlo (same inputs -> same result)
- Auto-load data.xlsx from same folder
- Mod column left-aligned, shows only first line of multiline names
- Filter restores full list when cleared
- Results shown bold + average Chaos Orbs needed
"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import pandas as pd
import numpy as np
import os
import hashlib
import math
import json

# INTERNAL CONSTANTS
DEFAULT_ITERATIONS = 2_000  # hidden from UI, used internally
BATCH = 1000  # update progress in batches


# -----------------------------
# Monte Carlo worker (thread)
# -----------------------------
class SimulationWorker(threading.Thread):
    def __init__(self, mods_df, thresholds, iterations, progress_callback, done_callback, stop_event, seed_int):
        super().__init__(daemon=True)
        # copy minimal, normalized DataFrame for thread-safety
        self.mods_df = mods_df.copy().reset_index(drop=True)
        self.thresholds = thresholds
        self.iterations = int(iterations)
        self.progress_callback = progress_callback
        self.done_callback = done_callback
        self.stop_event = stop_event
        self.seed_int = int(seed_int) & 0xFFFFFFFF

    def run(self):
        try:
            # Use a local Generator seeded deterministically for reproducible results
            rng = np.random.default_rng(self.seed_int)

            cols = ["quantity", "rarity", "pack_size", "currency", "scarabs", "maps"]
            thresholds_arr = np.array([self.thresholds[c] for c in cols], dtype=float)

            prefixes_df = self.mods_df[self.mods_df["type"] == "prefix"].reset_index(drop=True)
            suffixes_df = self.mods_df[self.mods_df["type"] == "suffix"].reset_index(drop=True)

            # Prepare weights arrays (if empty, we'll handle later)
            if len(prefixes_df) > 0:
                p_w = prefixes_df.get("weight", pd.Series(1, index=prefixes_df.index)).to_numpy(dtype=float)
                p_w = np.clip(p_w, 0.0, None)
                if p_w.sum() == 0:
                    p_w[:] = 1.0
                p_w = p_w / p_w.sum()
            else:
                p_w = None

            if len(suffixes_df) > 0:
                s_w = suffixes_df.get("weight", pd.Series(1, index=suffixes_df.index)).to_numpy(dtype=float)
                s_w = np.clip(s_w, 0.0, None)
                if s_w.sum() == 0:
                    s_w[:] = 1.0
                s_w = s_w / s_w.sum()
            else:
                s_w = None

            roll_sizes = np.array([4, 5, 6])
            roll_probs = np.array([8.0, 3.0, 1.0])
            roll_probs = roll_probs / roll_probs.sum()

            successes = 0
            completed = 0
            iters = self.iterations

            # Run simulation in batches to update progress
            while completed < iters and not self.stop_event.is_set():
                current = min(BATCH, iters - completed)
                # draw roll sizes (deterministic RNG)
                ks = rng.choice(roll_sizes, size=current, p=roll_probs)
                totals = np.zeros((current, len(cols)), dtype=float)

                for i, k in enumerate(ks):
                    # Determine random split prefix/suffix with max 3 each
                    max_prefix = min(3, k)
                    max_suffix = min(3, k)
                    possible_splits = [(p, k - p) for p in range(0, k + 1) if p <= max_prefix and (k - p) <= max_suffix]
                    # choose a split uniformly among valid splits
                    prefix_count, suffix_count = possible_splits[rng.integers(len(possible_splits))]

                    sum_vector = np.zeros(len(cols), dtype=float)

                    # sample prefixes
                    if prefix_count > 0 and len(prefixes_df) > 0:
                        sample_count = min(prefix_count, len(prefixes_df))
                        # choose without replacement using p_w (if sample_count == len -> take all)
                        if sample_count == len(prefixes_df):
                            sel_idx = np.arange(len(prefixes_df))
                        else:
                            sel_idx = rng.choice(len(prefixes_df), size=sample_count, replace=False, p=p_w)
                        # sum chosen prefix stats
                        sum_vector += prefixes_df.iloc[sel_idx][cols].sum(axis=0).to_numpy(dtype=float)

                    # sample suffixes
                    if suffix_count > 0 and len(suffixes_df) > 0:
                        sample_count = min(suffix_count, len(suffixes_df))
                        if sample_count == len(suffixes_df):
                            sel_idx = np.arange(len(suffixes_df))
                        else:
                            sel_idx = rng.choice(len(suffixes_df), size=sample_count, replace=False, p=s_w)
                        sum_vector += suffixes_df.iloc[sel_idx][cols].sum(axis=0).to_numpy(dtype=float)

                    totals[i, :] = sum_vector

                # count successes for this batch
                successes += int((totals >= thresholds_arr).all(axis=1).sum())
                completed += current

                # progress callback (0-100)
                if self.progress_callback:
                    pct = int(completed / iters * 100)
                    self.progress_callback(pct)

            # cancelled?
            if self.stop_event.is_set():
                if self.done_callback:
                    self.done_callback(None, cancelled=True)
                return

            probability = successes / float(iters) if iters > 0 else 0.0
            if self.done_callback:
                self.done_callback(probability, cancelled=False)

        except Exception as exc:
            if self.done_callback:
                self.done_callback(exc, error=True)


# -----------------------------
# Main Tkinter App
# -----------------------------
class ChaosOrbApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chaos Orb Outcome Estimator")
        self.geometry("1250x700")

        self.mods_df = None
        self.worker = None
        self.stop_event = threading.Event()
        self._build_ui()

        # Auto-load data.xlsx from same folder
        self.auto_load_excel()

    # -----------------------------
    # Auto-load Excel
    # -----------------------------
    def auto_load_excel(self):
        folder = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(folder, "data.xlsx")
        if os.path.exists(path):
            self.load_excel(path)
        else:
            self.status_var.set("Ready. data.xlsx not found in app folder.")

    # -----------------------------
    # Load Excel
    # -----------------------------
    def load_excel(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(title="Open mods Excel", filetypes=[("Excel files", "*.xlsx *.xls")])
        if not path:
            return
        try:
            df = pd.read_excel(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Excel:\n{e}")
            return

        # Required columns & defaults
        needed = ["mod", "type", "quantity", "rarity", "pack_size", "currency", "scarabs", "maps", "weight"]
        for c in needed:
            if c not in df.columns:
                df[c] = 1 if c == "weight" else 0

        # Normalize columns
        df["mod"] = df["mod"].astype(str)

        # Convert type: accept 1/2 or text; map to 'prefix'/'suffix'
        df["type"] = df["type"].astype(str).str.strip().str.lower()
        df["type"] = df["type"].replace({"1": "prefix", "2": "suffix", "prefix": "prefix", "suffix": "suffix"})
        # Any invalid entries default to prefix (safer)
        df.loc[~df["type"].isin(["prefix", "suffix"]), "type"] = "prefix"

        # Numeric coercion
        for c in ["quantity", "rarity", "pack_size", "currency", "scarabs", "maps", "weight"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1 if c == "weight" else 0)

        # Save normalized dataframe
        self.mods_df = df[needed].reset_index(drop=True)
        if "excluded" not in self.mods_df.columns:
            self.mods_df["excluded"] = False

        self.populate_tree()
        self.status_var.set(f"Loaded {len(df)} mods from {os.path.basename(path)}")

    # -----------------------------
    # Populate tree
    # -----------------------------
    def populate_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if self.mods_df is None:
            return
        for i, row in self.mods_df.iterrows():
            # Only show first line of multiline mod names
            mod_display = str(row["mod"]).splitlines()[0]
            values = (
                "", mod_display, row["type"],
                str(row["quantity"]), str(row["rarity"]), str(row["pack_size"]),
                str(row["currency"]), str(row["scarabs"]), str(row["maps"]), str(row["weight"])
            )
            self.tree.insert("", "end", iid=str(i), values=values)
        # bind double click after items are present
        self.tree.bind("<Double-1>", self._on_tree_double_click)

    def _on_tree_double_click(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        idx = int(item)
        self.mods_df.at[idx, "excluded"] = not self.mods_df.at[idx, "excluded"]
        curvals = list(self.tree.item(item, "values"))
        curvals[0] = "X" if self.mods_df.at[idx, "excluded"] else ""
        self.tree.item(item, values=curvals)

    # -----------------------------
    # Filter logic
    # -----------------------------
    def apply_filter(self):
        txt = self.filter_var.get().strip().lower()
        for iid in self.tree.get_children():
            modname = self.mods_df.loc[int(iid), "mod"].splitlines()[0].lower()
            if txt == "" or txt in modname:
                try:
                    self.tree.reattach(iid, "", "end")
                except Exception:
                    pass
            else:
                self.tree.detach(iid)

    def clear_exclusions(self):
        if self.mods_df is None:
            return
        self.mods_df["excluded"] = False
        for iid in self.tree.get_children():
            curvals = list(self.tree.item(iid, "values"))
            curvals[0] = ""
            self.tree.item(iid, values=curvals)

    def get_filtered_mods(self):
        if self.mods_df is None:
            return pd.DataFrame(columns=["mod", "type", "quantity", "rarity", "pack_size", "currency", "scarabs", "maps", "weight"])
        return self.mods_df[~self.mods_df["excluded"]].reset_index(drop=True)

    # -----------------------------
    # Build UI
    # -----------------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=8, pady=8)

        self.load_btn = ttk.Button(top, text="Load Mods Excel", command=self.load_excel)
        self.load_btn.pack(side="left")

        # Removed iterations & seed inputs per request (fixed internal iterations)
        self.save_btn = ttk.Button(top, text="Save Filtered Mods", command=self.save_filtered_mods)
        self.save_btn.pack(side="right")

        # Middle split
        middle = ttk.Panedwindow(self, orient="horizontal")
        middle.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        # Left: tree + filter
        left_frame = ttk.Frame(middle, width=800)
        middle.add(left_frame, weight=3)

        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(side="top", fill="x", pady=(0, 6))
        ttk.Label(filter_frame, text="Filter mods by name:").pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace("w", lambda *_: self.apply_filter())
        self.filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=30)
        self.filter_entry.pack(side="left", padx=(6, 0))
        self.clear_exclusions_btn = ttk.Button(filter_frame, text="Clear Exclusions", command=self.clear_exclusions)
        self.clear_exclusions_btn.pack(side="right")

        cols = ("Exclude", "Mod", "type", "quantity", "rarity", "pack_size", "currency", "scarabs", "maps", "weight")
        self.tree = ttk.Treeview(left_frame, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self.tree.heading(c, text=c)
            # Left-align mod text
            if c == "Mod":
                self.tree.column(c, width=360, anchor="w")
            else:
                self.tree.column(c, width=90, anchor="center")
        vsb = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Right frame: thresholds, run/cancel, results
        right_frame = ttk.Frame(middle, width=400)
        middle.add(right_frame, weight=1)

        th_group = ttk.LabelFrame(right_frame, text="Minimum thresholds (sum of affixes on roll)")
        th_group.pack(fill="x", padx=6, pady=6)
        self.threshold_vars = {}
        for name in ["currency", "scarabs", "maps", "quantity", "rarity", "pack_size"]:
            row = ttk.Frame(th_group)
            row.pack(fill="x", padx=6, pady=4)
            ttk.Label(row, text=f"{name}:").pack(side="left")
            v = tk.StringVar(value="0")
            ttk.Entry(row, textvariable=v, width=14).pack(side="right")
            self.threshold_vars[name] = v

        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill="x", padx=6, pady=6)
        self.run_btn = ttk.Button(btn_frame, text="Run Simulation", command=self.run_simulation)
        self.run_btn.pack(side="left", padx=(0, 6))
        self.cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.cancel_simulation, state="disabled")
        self.cancel_btn.pack(side="left")

        self.progress = ttk.Progressbar(right_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=6, pady=(6, 0))

        # Results area with bold font
        res_group = ttk.LabelFrame(right_frame, text="Results")
        res_group.pack(fill="both", expand=True, padx=6, pady=6)
        bold_font = tkfont.Font(size=11, weight="bold")
        self.result_label = tk.Label(res_group, text="No results yet.", wraplength=360, justify="left", font=bold_font, anchor="nw")
        self.result_label.pack(fill="both", expand=True, padx=8, pady=8)

        # status bar
        self.status_var = tk.StringVar(value="Ready.")
        status_label = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_label.pack(side="bottom", fill="x", padx=8, pady=6)

    # -----------------------------
    # Simulation flow
    # -----------------------------
    def run_simulation(self):
        if self.mods_df is None:
            messagebox.showwarning("No data", "Please load an Excel file before running.")
            return
        # read thresholds
        try:
            thresholds = {k: float(v.get()) for k, v in self.threshold_vars.items()}
        except Exception:
            messagebox.showwarning("Invalid thresholds", "Please enter numeric threshold values.")
            return

        df_filtered = self.get_filtered_mods()
        if df_filtered.empty:
            messagebox.showwarning("No mods", "All mods excluded or no mods available after filtering.")
            return

        # compute deterministic seed from filtered mods + thresholds
        seed_int = self._compute_deterministic_seed(df_filtered, thresholds)

        # disable UI
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress["value"] = 0
        self.status_var.set("Simulation running...")

        self.stop_event.clear()
        self.worker = SimulationWorker(df_filtered, thresholds, DEFAULT_ITERATIONS, self._on_progress, self._on_done, self.stop_event, seed_int)
        self.worker.start()

    def cancel_simulation(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.status_var.set("Cancelling...")

    def _on_progress(self, pct):
        # schedule on main thread
        self.after(0, lambda: self.progress.config(value=pct))

    def _on_done(self, result, cancelled=False, error=False):
        def ui_update():
            self.run_btn.config(state="normal")
            self.cancel_btn.config(state="disabled")
            if cancelled:
                self.status_var.set("Simulation cancelled.")
                messagebox.showinfo("Cancelled", "Simulation was cancelled.")
                return
            if error:
                self.status_var.set("Error during simulation")
                messagebox.showerror("Error", f"Simulation error:\n{result}")
                return
            prob = float(result)
            self.progress["value"] = 100
            self.status_var.set("Simulation finished.")
            # format probability as percent with 3 decimals
            prob_percent = prob * 100.0
            if prob > 0:
                avg_orbs = math.ceil(1.0 / prob)
            else:
                avg_orbs = "∞"
            self.result_label.config(text=f"Estimated probability per Chaos Orb roll: {prob_percent:.3f}%\nAverage Chaos Orbs Needed (expected): {avg_orbs}")
        self.after(0, ui_update)

    # -----------------------------
    # Utilities
    # -----------------------------
    def _compute_deterministic_seed(self, df: pd.DataFrame, thresholds: dict) -> int:
        """
        Create a deterministic integer seed from the filtered mods + thresholds.
        This ensures repeated runs with the same inputs produce identical results.
        """
        # build a compact, stable JSON representation:
        # include mod name, type, weight and numeric stat columns; sort rows for stability
        df_small = df[["mod", "type", "weight", "quantity", "rarity", "pack_size", "currency", "scarabs", "maps"]].copy()
        # round floats to limited precision for stability
        df_small[["weight", "quantity", "rarity", "pack_size", "currency", "scarabs", "maps"]] = df_small[
            ["weight", "quantity", "rarity", "pack_size", "currency", "scarabs", "maps"]
        ].round(6)
        # sort rows by mod name + type to make order-independent
        df_small = df_small.sort_values(by=["mod", "type"]).reset_index(drop=True)
        payload = {
            "mods": df_small.to_dict(orient="records"),
            "thresholds": {k: float(v) for k, v in sorted(thresholds.items())}
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        # hash to 32-bit int
        h = hashlib.md5(raw.encode("utf-8")).hexdigest()
        seed_int = int(h[:8], 16)  # take first 8 hex digits -> 32-bit
        return seed_int

    def save_filtered_mods(self):
        df = self.get_filtered_mods()
        if df.empty:
            messagebox.showinfo("No mods", "No mods to save (all excluded).")
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if not path:
            return
        try:
            df.to_excel(path, index=False)
            messagebox.showinfo("Saved", f"Saved {len(df)} mods to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")


# -----------------------------
# Entry point
# -----------------------------
def main():
    app = ChaosOrbApp()
    app.mainloop()


if __name__ == "__main__":
    main()
