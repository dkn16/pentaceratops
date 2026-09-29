#!/usr/bin/env python3
"""Download, FGP-preprocess, cache, and plot a TESS candidate.

Change TARGET below, or use --target / --candidate on the command line.
Run expensive preparation in a compute job; no FPP sampling is started here.
Install the package with its catalogs extra first: pip install -e '.[catalogs]'.
"""

TARGET = "TOI-4616.01"  # Alternatively "TIC 258796169" for a unique host signal.

if __name__ == "__main__":
    from pentaceratops.preprocessing.example_cli import main
    raise SystemExit(main("TESS", TARGET))
