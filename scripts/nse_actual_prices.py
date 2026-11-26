#!/usr/bin/env python3
"""Download the *actual traded* NSE close for every trading day in an existing
RiskLens price Excel, straight from NSE's official daily bhavcopy archive.

Why: Yahoo's `Close` is retroactively adjusted for splits AND for events it
does not list as splits (demergers, rights issues), so it never equals the
price people actually traded at. The bhavcopy is the exchange's own record,
so it needs no adjustment at all.

Two file formats exist: the legacy `cmDDMONYYYYbhav.csv.zip` (up to 5 Jul
2024) and the UDiFF `BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip` (from
8 Jul 2024). Files are cached so an interrupted run resumes.

Usage: python scripts/nse_actual_prices.py INPUT.xlsx OUTPUT.csv [CACHE_DIR]
"""
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

HOST = "https://nsearchives.nseindia.com/content"
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
HEADERS = {"User-Agent": "Mozilla/5.0"}
# NSE symbols that were renamed inside the 15-year window (Infosys: INFOSYSTCH -> INFY, mid-2011).
OLD_SYMBOLS = {"INFY": "INFOSYSTCH"}


def urls(d: pd.Timestamp) -> list[str]:
    old = f"{HOST}/historical/EQUITIES/{d.year}/{MON[d.month - 1]}/cm{d.day:02d}{MON[d.month - 1]}{d.year}bhav.csv.zip"
    new = f"{HOST}/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    return [new, old] if d >= pd.Timestamp("2024-07-08") else [old, new]


def fetch(d: pd.Timestamp, cache: Path) -> pd.Series | None:
    """Return {symbol: close} for one day (EQ series only), or None if unavailable."""
    f = cache / f"{d:%Y%m%d}.csv"
    if f.exists():
        s = pd.read_csv(f, index_col=0).iloc[:, 0]
        return s if len(s) else None
    for url in urls(d):
        for attempt in range(3):
            try:
                r = requests.get(url, headers=HEADERS, timeout=30)
            except requests.RequestException:
                time.sleep(1 + attempt)
                continue
            if r.status_code == 200 and r.content[:2] == b"PK":
                z = zipfile.ZipFile(io.BytesIO(r.content))
                df = pd.read_csv(z.open(z.namelist()[0]))
                if "TckrSymb" in df:
                    df = df.rename(columns={"TckrSymb": "SYMBOL", "SctySrs": "SERIES", "ClsPric": "CLOSE"})
                s = df[df["SERIES"].str.strip() == "EQ"].set_index("SYMBOL")["CLOSE"].astype(float)
                s.to_csv(f)
                return s
            if r.status_code == 404:
                break
            time.sleep(1 + attempt)
    f.write_text("SYMBOL,CLOSE\n")  # remember "no file for this date"
    return None


def main(src: Path, dst: Path, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    given = pd.read_excel(src, sheet_name=0 if len(sys.argv) < 5 else sys.argv[4], index_col=0, parse_dates=True)
    dates = pd.to_datetime(given.index).normalize()
    symbols = {t: t.removesuffix(".NS") for t in given.columns}

    with ThreadPoolExecutor(8) as ex:
        days = list(ex.map(lambda d: fetch(d, cache), dates))

    out = pd.DataFrame(index=dates, columns=list(symbols), dtype=float)
    missing = []
    for d, s in zip(dates, days):
        if s is None:
            missing.append(d.date())
            continue
        for t, sym in symbols.items():
            for name in (sym, OLD_SYMBOLS.get(sym)):
                if name in s.index:
                    out.loc[d, t] = s[name]
                    break
    out = out.drop(index=pd.to_datetime(missing))  # exchange closed (elections etc.) -- not trading days
    out.index.name = "Date"
    out.to_csv(dst)
    print(f"Wrote {dst}: {len(out)} days, {out.isna().any(axis=1).sum()} days with a gap; "
          f"dropped {len(missing)} dates with no bhavcopy (exchange closed)")
    if missing:
        print("No bhavcopy for:", missing[:20], "..." if len(missing) > 20 else "")


if __name__ == "__main__":
    cache = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("nse_cache")
    main(Path(sys.argv[1]), Path(sys.argv[2]), cache)
