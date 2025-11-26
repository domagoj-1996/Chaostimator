#!/usr/bin/env python3

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont
import pandas as pd
import numpy as np
import os
import math

# --- Config ---
DEFAULT_ITERATIONS = 2_000
BATCH = 200
FIXED_RNG_SEED = 1337

# --- Monte Carlo worker ---
class SimulationWorker(threading.Thread):
    def __init__(self, mods_df, thresholds, iterations, progress_callback, done_callback, stop_event):
        super().__init__(daemon=True)
        # thread-safe copy
        self.mods_df = mods_df.copy().reset_index(drop=True)
        self.thresholds = thresholds
        self.iterations = int(iterations)
        self.progress_callback = progress_callback
        self.done_callback = done_callback
        self.stop_event = stop_event

    def run(self):
        try:
            rng = np.random.default_rng(FIXED_RNG_SEED)

            cols = ["quantity", "rarity", "pack_size", "currency", "scarabs", "maps"]
            thresholds_arr = np.array([self.thresholds.get(c, 0.0) for c in cols], dtype=float)

            prefixes_df = self.mods_df[self.mods_df["type"] == "prefix"].reset_index(drop=True)
            suffixes_df = self.mods_df[self.mods_df["type"] == "suffix"].reset_index(drop=True)

            # prepare normalized weights per type
            if len(prefixes_df) > 0:
                p_w = prefixes_df["weight"].to_numpy(dtype=float)
                p_w = np.clip(p_w, 0.0, None)
                if p_w.sum() == 0.0:
                    p_w[:] = 1.0
                p_w = p_w / p_w.sum()
            else:
                p_w = None

            if len(suffixes_df) > 0:
                s_w = suffixes_df["weight"].to_numpy(dtype=float)
                s_w = np.clip(s_w, 0.0, None)
                if s_w.sum() == 0.0:
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

            while completed < iters and not self.stop_event.is_set():
                current = min(BATCH, iters - completed)
                ks = rng.choice(roll_sizes, size=current, p=roll_probs)
                totals = np.zeros((current, len(cols)), dtype=float)

                for i, k in enumerate(ks):
                    max_p = min(3, k)
                    max_s = min(3, k)
                    splits = [(p, k - p) for p in range(0, k + 1) if p <= max_p and (k - p) <= max_s]
                    prefix_count, suffix_count = splits[rng.integers(len(splits))]

                    sum_vec = np.zeros(len(cols), dtype=float)

                    # select prefixes
                    if prefix_count > 0 and len(prefixes_df) > 0:
                        take = min(prefix_count, len(prefixes_df))
                        if take == len(prefixes_df):
                            sel_idx = np.arange(len(prefixes_df))
                        else:
                            # weighted without replacement using probabilities p_w
                            sel_idx = rng.choice(len(prefixes_df), size=take, replace=False, p=p_w)
                        sum_vec += prefixes_df.iloc[sel_idx][cols].sum(axis=0).to_numpy(dtype=float)

                    # select suffixes
                    if suffix_count > 0 and len(suffixes_df) > 0:
                        take = min(suffix_count, len(suffixes_df))
                        if take == len(suffixes_df):
                            sel_idx = np.arange(len(suffixes_df))
                        else:
                            sel_idx = rng.choice(len(suffixes_df), size=take, replace=False, p=s_w)
                        sum_vec += suffixes_df.iloc[sel_idx][cols].sum(axis=0).to_numpy(dtype=float)

                    totals[i, :] = sum_vec

                successes += int((totals >= thresholds_arr).all(axis=1).sum())
                completed += current

                if self.progress_callback:
                    pct = int(completed / iters * 100)
                    self.progress_callback(pct)

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


# --- App ---
class ChaosOrbApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chaos Orb Outcome Estimator")
        self.geometry("1250x700")

        self.mods_df = None
        self.worker = None
        self.stop_event = threading.Event()

        self._build_ui()
        self.auto_load_excel()

    # --- Auto load ---
    def auto_load_excel(self):
        folder = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(folder, "data.xlsx")
        if os.path.exists(path):
            self.load_excel(path)
        else:
            self.status_var.set("Ready. data.xlsx not found in app folder.")

    # --- Load Excel ---
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

        # Normalize column names to make robust to small variations:
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")

        # Expected column names after normalization
        expected = ["mod", "type", "quantity", "rarity", "pack_size", "currency", "scarabs", "maps", "weight"]
        # If some missing, add defaults
        for col in expected:
            if col not in df.columns:
                df[col] = 0 if col != "weight" else 1

        # Ensure mod is string
        df["mod"] = df["mod"].astype(str)

        # Map type column: accept numeric 1/2 or textual prefix/suffix
        def map_type(v):
            try:
                # try numeric
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
            # default fallback
            return "prefix"

        df["type"] = df["type"].apply(map_type)

        # numeric columns coerced
        for c in ["quantity", "rarity", "pack_size", "currency", "scarabs", "maps", "weight"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1 if c == "weight" else 0)

        # final df with expected column names
        self.mods_df = df[expected].reset_index(drop=True)
        if "excluded" not in self.mods_df.columns:
            self.mods_df["excluded"] = False

        self.populate_tree()
        self.status_var.set(f"Loaded {len(self.mods_df)} mods from {os.path.basename(path)}")

    # --- Populate tree (full) ---
    def populate_tree(self):
        # clear
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        if self.mods_df is None:
            return
        for i, row in self.mods_df.iterrows():
            mod_display = str(row["mod"]).splitlines()[0]
            excl = "X" if bool(row.get("excluded", False)) else ""
            values = (
                excl,
                mod_display,
                row["type"],
                f"{row['quantity']}",
                f"{row['rarity']}",
                f"{row['pack_size']}",
                f"{row['currency']}",
                f"{row['scarabs']}",
                f"{row['maps']}",
                f"{row['weight']}",
            )
            self.tree.insert("", "end", iid=str(i), values=values)

    # --- Double click toggle exclude ---
    def _on_tree_double_click(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        idx = int(item)
        self.mods_df.at[idx, "excluded"] = not self.mods_df.at[idx, "excluded"]
        # refresh display for that row
        excl = "X" if self.mods_df.at[idx, "excluded"] else ""
        curvals = list(self.tree.item(item, "values"))
        curvals[0] = excl
        self.tree.item(item, values=curvals)

    # --- Filter ---
    def apply_filter(self):
        txt = self.filter_var.get().strip().lower()

        # if empty -> restore full list
        if txt == "":
            self.populate_tree()
            return

        # build filtered view
        # clear tree
        for iid in list(self.tree.get_children()):
            self.tree.delete(iid)

        for i, row in self.mods_df.iterrows():
            name = str(row["mod"]).splitlines()[0].lower()
            if txt in name:
                excl = "X" if bool(row.get("excluded", False)) else ""
                values = (
                    excl,
                    str(row["mod"]).splitlines()[0],
                    row["type"],
                    f"{row['quantity']}",
                    f"{row['rarity']}",
                    f"{row['pack_size']}",
                    f"{row['currency']}",
                    f"{row['scarabs']}",
                    f"{row['maps']}",
                    f"{row['weight']}",
                )
                self.tree.insert("", "end", iid=str(i), values=values)

    # --- Sorting helper ---
    def sort_tree(self, col, descending=False):
        # gather values
        data = []
        for child in self.tree.get_children():
            val = self.tree.set(child, col)
            data.append((val, child))

        # attempt numeric sort
        def try_float(x):
            try:
                return float(x)
            except Exception:
                return None

        numeric = True
        for v, _ in data:
            if try_float(v) is None:
                numeric = False
                break

        if numeric:
            data.sort(key=lambda t: float(t[0]), reverse=descending)
        else:
            data.sort(key=lambda t: t[0].lower(), reverse=descending)

        # rearrange
        for index, (_, iid) in enumerate(data):
            self.tree.move(iid, "", index)

        # toggle next time
        self.tree.heading(col, command=lambda: self.sort_tree(col, not descending))

    # --- Clear exclusions ---
    def clear_exclusions(self):
        if self.mods_df is None:
            return
        self.mods_df["excluded"] = False
        self.populate_tree()

    # --- Get filtered mods as DataFrame ---
    def get_filtered_mods(self):
        if self.mods_df is None:
            return pd.DataFrame(columns=["mod", "type", "quantity", "rarity", "pack_size", "currency", "scarabs", "maps", "weight"])
        return self.mods_df[~self.mods_df["excluded"]].reset_index(drop=True)

    # --- UI build ---
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
        left_frame = ttk.Frame(middle, width=800)
        middle.add(left_frame, weight=3)

        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(side="top", fill="x", pady=(0, 6))
        ttk.Label(filter_frame, text="Filter mods by name:").pack(side="left")
        self.filter_var = tk.StringVar()
        # use trace_add for newer tkinter
        self.filter_var.trace_add("write", lambda *_: self.apply_filter())
        self.filter_entry = ttk.Entry(filter_frame, textvariable=self.filter_var, width=30)
        self.filter_entry.pack(side="left", padx=(6, 0))
        self.clear_exclusions_btn = ttk.Button(filter_frame, text="Clear Exclusions", command=self.clear_exclusions)
        self.clear_exclusions_btn.pack(side="right")

        cols = ("Exclude", "Mod", "type", "quantity", "rarity", "pack_size", "currency", "scarabs", "maps", "weight")
        self.tree = ttk.Treeview(left_frame, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            # set heading with sort command (click to sort)
            self.tree.heading(c, text=c, command=lambda _c=c: self.sort_tree(_c, False))
            if c == "Mod":
                self.tree.column(c, width=360, anchor="w")
            else:
                self.tree.column(c, width=90, anchor="center")

        vsb = ttk.Scrollbar(left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # right: thresholds + controls + results
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

        res_group = ttk.LabelFrame(right_frame, text="Results")
        res_group.pack(fill="both", expand=True, padx=6, pady=6)
        bold_font = tkfont.Font(size=12, weight="bold")
        self.result_label = tk.Label(res_group, text="No results yet.", wraplength=360, justify="left", font=bold_font, anchor="nw")
        self.result_label.pack(fill="both", expand=True, padx=8, pady=8)

        self.status_var = tk.StringVar(value="Ready.")
        status_label = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_label.pack(side="bottom", fill="x", padx=8, pady=6)

        # bind double click now (tree items will be inserted later)
        self.tree.bind("<Double-1>", self._on_tree_double_click)

    # --- Run simulation ---
    def run_simulation(self):
        if self.mods_df is None:
            messagebox.showwarning("No data", "Please load an Excel file before running.")
            return

        # parse thresholds
        try:
            thresholds = {k: float(v.get()) for k, v in self.threshold_vars.items()}
        except Exception:
            messagebox.showwarning("Invalid thresholds", "Please enter numeric threshold values.")
            return

        df_filtered = self.get_filtered_mods()
        if df_filtered.empty:
            messagebox.showwarning("No mods", "All mods excluded or no mods available after filtering.")
            return

        # disable UI controls
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.progress["value"] = 0
        self.status_var.set("Simulation running...")

        self.stop_event.clear()
        self.worker = SimulationWorker(df_filtered, thresholds, DEFAULT_ITERATIONS, self._on_progress, self._on_done, self.stop_event)
        self.worker.start()

    def cancel_simulation(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.status_var.set("Cancelling...")

    def _on_progress(self, pct):
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
            prob_pct = prob * 100.0
            if prob > 0:
                avg = math.ceil(1.0 / prob)
            else:
                avg = "∞"
            self.progress["value"] = 100
            self.status_var.set("Simulation finished.")
            self.result_label.config(text=f"Estimated probability per Chaos Orb roll: {prob_pct:.3f}%\nAverage Chaos Orbs Needed (expected): {avg}")
        self.after(0, ui_update)

    # --- Save filtered mods ---
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


# --- Entry point ---
def main():
    app = ChaosOrbApp()
    app.mainloop()


if __name__ == "__main__":
    main()
