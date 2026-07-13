import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import BENCHMARK_TICKER, END_DATE, START_DATE, UNIVERSE_NAME
from data.benchmark import load_benchmark
from data.prices import add_returns, coverage_report, load_price_panel
from data.universe import get_universe


def main():
    parser = argparse.ArgumentParser(description="Build and cache the Phase 0 price panel.")
    parser.add_argument("--universe", default=UNIVERSE_NAME)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--benchmark", default=BENCHMARK_TICKER)
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached parquet and re-download."
    )
    args = parser.parse_args()

    print(
        "NOTE: universe = current constituents projected back over the lookback "
        "window, so removed/delisted/acquired tickers are absent (survivorship bias). "
        "See NOTES.md."
    )

    tickers = get_universe(args.universe)
    print(f"Universe '{args.universe}': {len(tickers)} tickers")

    panel = load_price_panel(tickers, start=args.start, end=args.end, use_cache=not args.refresh)
    panel = add_returns(panel)
    print(f"\nPrice panel: {len(panel)} rows, {panel['Ticker'].nunique()} tickers")

    report = coverage_report(panel)
    print("\nCoverage report (tickers with fewest rows first):")
    print(report.to_string())

    benchmark = load_benchmark(args.benchmark, start=args.start, end=args.end, use_cache=not args.refresh)
    print(f"\nBenchmark ({args.benchmark}): {len(benchmark)} rows, "
          f"{benchmark['Date'].min()} to {benchmark['Date'].max()}")


if __name__ == "__main__":
    main()
