#!/usr/bin/env python3
"""Download, FGP-preprocess, cache, and plot a Kepler candidate.

Change TARGET below, or use --target / --candidate on the command line.
Use KIC/KOI identifiers, not TIC IDs. Epoch overrides use full BJD.
Install the package with its catalogs extra first: pip install -e '.[catalogs]'.
"""

TARGET = "KIC 8758204"  # If this host has multiple KOIs, select one with --candidate.

if __name__ == "__main__":
    from pentaceratops.preprocessing.example_cli import main
    raise SystemExit(main("Kepler", TARGET))
