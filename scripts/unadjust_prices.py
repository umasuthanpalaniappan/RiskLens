#!/usr/bin/env python3
"""Reverse Yahoo's split/bonus adjustment to recover the actual traded close.

Yahoo's `Close` (even with auto_adjust=False) is retroactively adjusted for
stock splits and bonus issues -- only dividends are left alone. So an
ICICI Bank close of Rs 1,140 on 3 Jan 2011 shows up as Rs 208. This script:

  1. reads the existing RiskLens Excel export (dates + tickers),
  2. fetches every split / bonus event per ticker from Yahoo
     (Ticker.splits, ratio = new shares per old share; a 1:10 bonus is 1.1),
  3. multiplies each close by the product of all ratios whose ex-date is
     AFTER that day -- i.e. undoes the adjustment backwards from each ex-date,
  4. cross-checks the result against NSE's official closes (from
     scripts/nse_actual_prices.py) and writes a new Excel.

Splits alone are NOT enough for every stock: Yahoo also bakes demergers and
rights issues (Reliance/Jio Financial, ITC hotels, Bharti, Tata Steel) into
`Close` without listing them as splits. The NSE closes are therefore the
primary sheet; the split-reversed Yahoo series is kept for comparison.

Usage: python scripts/unadjust_prices.py INPUT.xlsx OUTPUT.xlsx NSE_ACTUAL.csv
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

SHEET = "Prices (Raw)"


def split_factors(index: pd.DatetimeIndex, splits: pd.Series) -> pd.Series:
    """Per-day multiplier = product of split ratios with ex-date > that day."""
    factor = pd.Series(1.0, index=index)
    for ex_date, ratio in splits.items():
        ex = pd.Timestamp(ex_date).tz_localize(None).normalize()
        factor[index < ex] *= float(ratio)
    return factor


def main(src: Path, dst: Path, nse_csv: Path) -> None:
    given = pd.read_excel(src, sheet_name=SHEET, index_col=0, parse_dates=True)
    given.index = pd.to_datetime(given.index).normalize()
    tickers = list(given.columns)

    # Full-precision closes (the Excel is rounded to 2 dp, which x5.5 would amplify).
    raw = yf.download(
        tickers,
        start=given.index.min().strftime("%Y-%m-%d"),
        end=(given.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    out = pd.DataFrame(index=given.index, columns=tickers, dtype=float)
    split_rows = []
    for t in tickers:
        close = raw[t]["Close"].copy()
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        close = close.reindex(given.index)
        splits = yf.Ticker(t).splits
        out[t] = close * split_factors(given.index, splits)
        for d, r in splits.items():
            d = pd.Timestamp(d).tz_localize(None).normalize()
            if given.index.min() <= d <= given.index.max():
                split_rows.append({"Ticker": t, "Ex-date": d.date(), "Ratio (new per old)": float(r),
                                   "Cumulative factor on first day": float(split_factors(given.index[:1], splits).iloc[0])})
    out = out.round(2)
    nse = pd.read_csv(nse_csv, index_col=0, parse_dates=True).round(2)
    out = out.reindex(nse.index)  # NSE was shut on a few dates Yahoo still has rows for
    out.index.name = nse.index.name = "Date"

    off = ((out / nse) - 1).abs() > 0.005
    verdict = {
        t: ("matches NSE" if off[t].sum() <= 10 else "Yahoo also adjusted for a demerger/rights issue not in its split list")
        for t in tickers
    }
    check = pd.DataFrame({
        "Jan-2011 close, Yahoo (split-adjusted)": given.iloc[0],
        "Jan-2011 close, split-reversed": out.iloc[0],
        "Jan-2011 close, NSE actual": nse.iloc[0],
        "Days differing from NSE by >0.5%": off.sum(),
        "Verdict": pd.Series(verdict),
    })
    check.index.name = "Ticker"
    splits_df = pd.DataFrame(split_rows).sort_values(["Ticker", "Ex-date"])
    readme = pd.DataFrame({"RiskLens - actual traded NSE close, 18 stocks": [
        f"Sheet 'Actual Traded Close': NSE's official daily close (bhavcopy), {len(nse)} trading days, "
        f"{nse.index.min().date()} to {nse.index.max().date()}. No split, bonus, demerger or dividend adjustment.",
        "This is the price a buyer actually paid on that date -- e.g. ICICIBANK 3 Jan 2011 = 1144.85, not 208.15.",
        "Sheet 'Split-reversed Yahoo': Yahoo close x product of later split/bonus ratios (the reverse-engineering asked for).",
        "It equals NSE except for Reliance, ITC, Bharti Airtel and Tata Steel, where Yahoo also adjusted for demergers/rights issues.",
        "Sheet 'Splits & Bonus Used': every split/bonus Yahoo lists in the window (ratio = new shares per old; 1:10 bonus = 1.1).",
        "Dates the NSE was closed (24 Apr 2014, 15 Oct 2014, 15 Jan 2026) are omitted.",
    ]})
    with pd.ExcelWriter(dst, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        nse.to_excel(xw, sheet_name="Actual Traded Close")
        out.to_excel(xw, sheet_name="Split-reversed Yahoo")
        check.to_excel(xw, sheet_name="Verification")
        splits_df.to_excel(xw, sheet_name="Splits & Bonus Used", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            ws.column_dimensions["A"].width = 16
            for col in "BCDEFGHIJKLMNOPQRS":
                ws.column_dimensions[col].width = 15
        xw.book["README"].column_dimensions["A"].width = 150
        xw.book["Verification"].column_dimensions["F"].width = 70
        for c in "ABCDE":
            xw.book["Verification"].column_dimensions[c].width = 34
        for sh in ("Actual Traded Close", "Split-reversed Yahoo"):
            for row in xw.book[sh].iter_rows(min_row=2, max_col=1):
                row[0].number_format = "yyyy-mm-dd"
    print(f"Wrote {dst}")
    print(check.to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
