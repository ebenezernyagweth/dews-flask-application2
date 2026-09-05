#!/usr/bin/env python3
"""
Rebuild results/ history from app_copy at the APRIL-RUN state.

The April run (25 May, last observed month = April) was the last estimation
with complete geospatial coverage. It produced:
    h=1 -> 2026-05 forecast
    h=2 -> 2026-06 forecast
    h=3 -> 2026-07 forecast
and actuals through April (May actuals were backfilled later; they are real
MUAC data and are kept).

app_copy also contains rows added by the June and July runs, which ran during
the billing outage on a 46-feature dataset. Those are dropped by truncating
each horizon at the April run's own forecast month.

Dry run by default. Pass --apply to write into results/.
"""

import argparse
import os
import glob
import shutil
import pandas as pd

APP_COPY = os.path.expanduser(
    "~/Desktop/DewsDB/app_copy/Kenya_MUAC_NDMA_implementation/results")
RESULTS = os.path.join(os.getcwd(), "results")
BACKUP = os.path.join(RESULTS, "prev_history_backup")

OUTCOMES = ["wasting_smoothed", "wasting_risk_smoothed"]

# April run: last observed = April, so h -> April + h
CUTOFF = {1: "2026-05-01", 2: "2026-06-01", 3: "2026-07-01"}


def find_one(folder, outcome, h):
    hits = glob.glob(os.path.join(folder, f"{outcome}_pred_hb_{h}_36m_*.csv"))
    hits = [p for p in hits if not p.endswith("_future_months.csv")]
    if len(hits) != 1:
        print(f"    !! {len(hits)} candidates for {outcome} h={h} in {folder}")
        return None
    return hits[0]


def build(outcome, h):
    print(f"\n  --- {outcome}  h={h} ---")
    src = find_one(APP_COPY, outcome, h)
    if src is None:
        return None

    d = pd.read_csv(src)
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
    d["time_period"] = pd.to_datetime(d["time_period"], errors="coerce")

    cut = pd.Timestamp(CUTOFF[h])
    dropped = d[d.time_period > cut]
    kept = d[d.time_period <= cut].copy()

    print(f"    source  : {os.path.basename(src)}")
    print(f"    rows    : {len(d)} -> {len(kept)}  (dropped {len(dropped)})")
    if len(dropped):
        months = sorted(dropped.time_period.dt.strftime("%Y-%m").unique())
        print(f"    removed : {months}   <- outage-era runs")

    obs = kept[outcome].notna().sum() if outcome in kept.columns else 0
    last_obs = (kept.loc[kept[outcome].notna(), "time_period"].max()
                if obs else pd.NaT)
    months = kept.time_period.dt.to_period("M")
    expected = pd.period_range(months.min(), months.max(), freq="M")
    gaps = sorted(set(expected) - set(months.unique()))

    print(f"    range   : {kept.time_period.min():%Y-%m} -> "
          f"{kept.time_period.max():%Y-%m}")
    print(f"    actuals : {obs} rows, through {last_obs:%Y-%m}"
          if obs else "    actuals : NONE")
    print(f"    gaps    : {[str(g) for g in gaps] if gaps else 'none'}")

    lo, hi = kept.time_period.min(), kept.time_period.max()
    name = (f"{outcome}_pred_hb_{h}_36m_"
            f"{lo.year}_{lo.month:02d}_to_{hi.year}_{hi.month:02d}.csv")
    return name, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    print(f"APP_COPY : {APP_COPY}")
    print(f"RESULTS  : {RESULTS}")
    if not os.path.isdir(APP_COPY):
        print("           ^ NOT FOUND - edit APP_COPY at the top")
        return

    outputs = []
    for outcome in OUTCOMES:
        for h in (1, 2, 3):
            got = build(outcome, h)
            if got:
                outputs.append(got)

    print("\n" + "=" * 68)
    if not args.apply:
        print("DRY RUN - nothing written. Would create in results/:")
        for name, df in outputs:
            print(f"  {name}   ({len(df)} rows)")
        print("\nRe-run with --apply to write.")
        return

    if len(outputs) != 6:
        print(f"Only {len(outputs)} of 6 built - not writing. Fix the source first.")
        return

    os.makedirs(BACKUP, exist_ok=True)
    for f in glob.glob(os.path.join(RESULTS, "*_pred_hb_*_36m_*.csv")):
        shutil.move(f, os.path.join(BACKUP, os.path.basename(f)))
        print(f"  backed up {os.path.basename(f)}")

    for name, df in outputs:
        df.to_csv(os.path.join(RESULTS, name), index=False)
        print(f"  wrote {name}  ({len(df)} rows)")
    print(f"\nPrevious history moved to {BACKUP}")


if __name__ == "__main__":
    main()
