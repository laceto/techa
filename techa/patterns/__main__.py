"""CLI entry point: python -m techa.patterns TICKER [START] [END] [show|save]

Examples
--------
python -m techa.patterns SIE.DE
python -m techa.patterns SIE.DE 2020-01-01 2024-01-01
python -m techa.patterns SIE.DE 2020-01-01 2024-01-01 save
"""

import sys

import yfinance as yf

from techa.patterns import explore_patterns


def main() -> None:
    args = sys.argv[1:]
    symbol = args[0] if args else "SIE.DE"
    start = args[1] if len(args) > 1 else "2020-01-01"
    end = args[2] if len(args) > 2 else "2024-01-01"
    output = args[3] if len(args) > 3 else "show"

    print(f"Downloading {symbol} ({start} → {end})...")
    data = yf.download(symbol, start=start, end=end, multi_level_index=False)
    if data.empty:
        print(f"No data returned for {symbol}. Check the ticker and date range.")
        sys.exit(1)

    explore_patterns(data, symbol=symbol, output=output)


if __name__ == "__main__":
    main()
