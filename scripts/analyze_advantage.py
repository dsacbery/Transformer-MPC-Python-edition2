from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trans_mpc.advantage_analysis import analyze_transformer_advantage, save_advantage_figures


def main() -> None:
    log_path = ROOT / "outputs" / "logs" / "experiment_log.csv"
    table_dir = ROOT / "outputs" / "tables"
    figure_dir = ROOT / "outputs" / "figures"
    log_df = pd.read_csv(log_path)
    summary_df, intervals_df = analyze_transformer_advantage(log_df)
    summary_df.to_csv(table_dir / "advantage_summary.csv", index=False)
    intervals_df.to_csv(table_dir / "advantage_intervals.csv", index=False)
    save_advantage_figures(log_df, summary_df, figure_dir)
    print(f"saved {table_dir / 'advantage_summary.csv'}")
    print(f"saved {table_dir / 'advantage_intervals.csv'}")
    print(f"saved {figure_dir / 'advantage_composite_score.png'}")
    print(f"saved {figure_dir / 'advantage_percentage_bars.png'}")


if __name__ == "__main__":
    main()
