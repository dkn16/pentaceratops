# Candidate download and preparation

These examples start from a TIC/TOI or KIC/KOI identifier. Edit `TARGET` near
the top of `tess_candidate.py` or `kepler_candidate.py`, or use `--target`.
They download and preprocess photometry; **they do not calculate an FPP**.

Once the preparation and stellar-field object are available, call:

```python
from pentaceratops.evidence import evidence
results = evidence(prepared, target=target, likelihood="fourier", eb_eta=0.1)
```

`prepared` is the returned preparation object or its saved NPZ path. Set
`likelihood="real"` for Real evidence. See the [Python interface](../docs/evidence_api.md)
for arguments and results. The optional command-line entry point remains
`pentaceratops evidence target.json`, with [evidence.json](evidence.json) as a
configuration template.

## Quick start

Run from the package checkout after installing the catalogue dependencies in
your development environment:

```bash
python -m pip install -e '.[catalogs]'
export PENTACERATOPS_CACHE="$PSCRATCH/pentaceratops/candidate_examples"

# Cheap ephemeris check, then optional download-only prefetch.
python examples/tess_candidate.py --target TOI-4616.01 --stage resolve
python examples/tess_candidate.py --target TOI-4616.01 --stage download

# Default stage: download if necessary, detrend, fold, save, and plot.
python examples/tess_candidate.py --target TOI-4616.01
python examples/kepler_candidate.py --target 'KIC 8758204'
```

Without an editable installation, add `PYTHONPATH=src` when running from this
checkout. For a first download test, restrict `--sectors 17` or `--quarters 4`.
A single quarter need not contain enough transits to prepare complete even,
odd, and secondary windows; downloading it can still succeed.

Run long real-data FGP preparation in Slurm. With the environment activated,
submit from the checkout, setting your own account and machine constraints:

```bash
sbatch -A YOUR_ACCOUNT -C cpu -q debug examples/prepare.slurm tess \
  --target TOI-4616.01 --sectors 17 --plot-dir "$HOME/pentaceratops_plots"
```

This template processes one target with one CPU and numerical-library thread.
For population work, use separate processes, with parallelism sized to the
allocation, rather than concurrent threads inside an engine.

## Ephemerides: NASA by default

The default source is NASA Exoplanet Archive: the `toi` table for TESS and
`cumulative` KOI table for Kepler. TIC/KIC identify hosts, not unique planets.
If a host has multiple rows, choose its signal explicitly:

```bash
python examples/tess_candidate.py --target 'TIC 150428135' --list-candidates
python examples/tess_candidate.py --target 'TIC 150428135' --candidate TOI-700.02
```

For TESS only, select Bayesian Exoplanets explicitly:

```bash
python examples/tess_candidate.py --target TOI-4616.01 --catalog-source bayesian
# Or pin your own local copy of the site's tois.csv:
python examples/tess_candidate.py --target TOI-4616.01 \
  --catalog-source bayesian --catalog-file /path/to/tois.csv
```

| Source | Period | Absolute transit epoch | Transit duration |
| --- | --- | --- | --- |
| NASA TOI | `pl_orbper` in days | `pl_tranmid` in full BJD | `pl_trandurh` in hours |
| NASA KOI | `koi_period` in days | `koi_time0bk` + 2454833 | `koi_duration` in hours |
| Bayesian TESS | `Period` in days | `Epoch` + 2457000 | `Duration` in days, converted to hours |

The Bayesian catalogue's `Phase` field is **not** used as an absolute epoch;
`Tau` is not substituted for its published `Duration`. Missing required
ephemerides cause an error, not a silent fallback to a different catalogue.
The Bayesian table does not supply the same depth field as NASA, so no depth
is fabricated or silently fetched from NASA when that source is selected.
Selecting a catalogue row does not assert that it passed validation or false-
alarm tests; a selected Bayesian row failing its tests emits a warning.

Sources: [NASA TOI columns](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html),
[NASA KOI columns](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html),
[Bayesian Exoplanets TESS catalogue](https://bayesianexoplanets.github.io/#tess).

Explicit `--period DAYS`, `--epoch-bjd FULL_BJD`, `--duration-hours HOURS`,
and `--depth-ppm PPM` override catalogue values. All supplied epochs must be
**full BJD**, not BTJD/BKJD. A TIC/KIC absent from the selected catalogue can
be used with all three required ephemeris overrides; arbitrary star names
and confirmed-planet name resolution are not implemented. Phase is derived
from the saved period and absolute epoch, not an independently guessed offset.

NASA companions with candidate/planet dispositions are masked automatically.
With the Bayesian source, other signals on the same TIC are masked, including
flagged signals, to avoid contaminating the selected transit. Masking depends
on catalogue completeness. Add an uncatalogued companion with repeatable
`--mask-planet PERIOD,FULL_BJD,DURATION_HOURS`. Incomplete companion ephemerides
raise an error instead of silently omitting the mask. `--keep-other-planets`
disables automatic companion masks for explicit diagnostic comparisons.

## What preparation does

1. Pin catalogue responses and resolve the target and companion ephemerides.
2. Select one official PDCSAP product per sector/quarter before downloading.
   TESS prefers SPOC, then TESS-SPOC; 20-second data require an explicit
   `--cadence-seconds 20`. Kepler defaults to long cadence. Apply Lightkurve's
   default quality filter and check the downloaded FITS host and segment IDs.
3. Read timestamps explicitly as BJD TDB. Normalize each product and restore
   its aperture-frame flux and errors once using `CROWDSAP`, before FGP fitting.
4. Mask other planets and bin onto regular per-segment grids. Default binning
   is 10 minutes for TESS and 30 minutes for Kepler, never finer than the
   slowest selected product. Missing observations remain missing.
5. Fit a separate iterative Fourier MAP GP/PSD in each sector/quarter, using
   only observations outside protected windows. The primary/secondary window
   half-width defaults to four durations, capped strictly below P/4. Protection
   uses the **same bin assignments as the final folded likelihood windows**.
   The initializer uses optional `celerite` if available; otherwise it warns
   and uses smoothed interpolation of training observations only.
6. Form `1 + (Gaussianized variation - FGP variation) / flux baseline`.
   Protected science observations are not Gaussianized as outliers. Fold
   even, odd, and phase-0.5 secondary windows with inverse-variance weights.
   The combined primary is a diagnostic, not an extra independent likelihood
   term alongside even/odd. Plot all four windows and each segment's trend.

Insufficient training data or empty folded bins cause explicit errors. No
interpolated gap values enter the folded observations. The example does not
automatically shrink protection or invent missing transit samples. Adjust
sector coverage, `--bin-minutes`, or `--window-durations` explicitly and inspect
the resulting plots. Phase-0.5 secondary placement is an assumption, not an
eccentric-secondary search.

The common bin width is saved as an approximate rectangular exposure duration.
This is not an exact mixed-cadence or irregular-bin integration response.
Window error bars are propagated measurement errors only, **not FGP posterior
uncertainties**. These examples are not a claim of exact equivalence to the
historical TESS benchmark or custom Kepler HZ preprocessing.

## Storage and replay

Bulk caches use `--cache-dir`, then `PENTACERATOPS_CACHE`, then a subdirectory
of `PSCRATCH`/`SCRATCH`, or finally the system temporary directory. Matplotlib,
Numba, and XDG caches default beneath it unless already configured. Plots
default to `plots/candidate_examples` in the working directory; set
`--plot-dir "$HOME/pentaceratops_plots"` to keep them in home.

```text
cache/
  catalog/     Pinned NASA responses or Bayesian table + hashes/source metadata
  lightkurve/  Downloaded FITS products
  native/      Selected, aperture-restored photometry + product provenance
  prepared/    Resolved settings, folded windows, segment arrays and MAP state
  runtime/     Plotting/compilation cache defaults
```

Native caches depend on target/product selection; prepared cache names include
ephemerides, source, masks, settings, and code fingerprints. Catalogue responses
and downloads are not implicitly refreshed. Use a new cache directory for a
fresh catalogue/product snapshot. `--offline` requires matching existing
catalogue and native caches; it can still compute a new preparation locally.

Prepared bundles use the pickle-free `RunResult` format:

```python
from pentaceratops import RunResult
saved = RunResult.load("/path/to/cache/prepared/target_hash.npz")
windows = saved.output["windows"]  # time_even, flux_even, err_even, etc.
segments = saved.output["segments"]
settings = saved.metadata["settings"]
```

Each segment retains its grid, native mask, protected/training masks, cleaned
flux, MAP trend, final MAP fit input, stellar PSD and white variance. The
catalogue records, product hashes, and normalization/restoration factors are
also saved. These sector PSDs are **not** prepared full/even/odd Fourier-
likelihood inputs. To redraw saved plots without a network query or FGP fit:

```bash
python examples/tess_candidate.py --plot-only /path/to/prepared/target_hash.npz \
  --plot-dir "$HOME/pentaceratops_plots"
```

For advanced Python use, `pentaceratops.preprocessing.candidate.prepare_candidate`
accepts the same main choices (`source="nasa"` by default), returning a
`RunResult` and its path. `preprocess_blocks` is the filesystem/network-free
array entry point. `--help` lists all CLI controls without querying catalogues.

Saved bundles explicitly mark `likelihood_ready=False`: they contain MAP
preparation rather than a likelihood covariance. `pentaceratops evidence`
finishes covariance propagation from the saved fit state and calls the optimized
evidence API, without repeating detrending or PSD estimation. Stellar-field and
companion-population inputs must still be supplied explicitly. Direct low-level
dispatchers do not perform this conversion automatically.
