# Preparing the stellar-field target

Both CSVs can be obtained directly from Python. `prepare_target` returns the
`target` needed by `evidence` and writes the field files for later reuse:

```python
from pentaceratops import prepare_target
from pentaceratops.preprocessing.candidate import prepare_candidate
from pentaceratops.evidence import evidence

prepared, prepared_path = prepare_candidate("TOI-700.02", cache_dir="/path/to/cache")
target = prepare_target(prepared, output_dir="/path/to/new_field")
results = evidence(prepared, target=target, likelihood="fourier", eb_eta=0.1)
```

`prepared` holds photometry and its saved noise model. `target` holds the host
star, nearby sources, aperture dilution and background population. Preparation
of a new stellar field currently supports **TESS**. Install the catalog extra
with `python -m pip install -e '.[catalogs]'` from the checkout; new queries need
network access. Use a compute allocation for light-curve preprocessing.

If the light curve is already prepared, skip `prepare_candidate`: pass its
`RunResult` or saved NPZ path directly to `prepare_target`.

## What the helper does

1. Reads the TIC host ID and science FITS products from the preparation bundle.
2. Queries TIC for the host and nearby stars, places the host first, and uses
   the actual science-file apertures/WCS to calculate dilution and eligibility.
3. Queries TRILEGAL at the host's coordinates and waits for the population.
4. Saves `stars.csv`, `trilegal.csv`, and `field.json` in `output_dir`.

`field.json` records the candidate, input FITS hashes, per-sector aperture
fractions, population provenance and CSV hashes. Access the returned table as
`target.stars`, its filename as `target.stars_path`, and the population filename
as `target.trilegal_fname`. No CSV export or configuration file is required to
pass this object to `evidence`.

The original downloaded FITS files must be available at their saved paths.
The dilution calculation uses the established 0.75-pixel Gaussian PSF and equal
sector weighting; see [aperture geometry](aperture-geometry.md). The saved
`windows["transit_depth"]` is already in the aperture flux frame and is used
without another CROWDSAP correction. Constructing a catalog `Target(...)`
alone does not calculate these dilution quantities.

The directory must be new. Existing fields are never overwritten. A failed
catalog, stellar-parameter, geometry or population check does not publish a
completed field. There is no fallback that silently omits background scenarios.

## Reuse without network access

```python
from pentaceratops import RunResult, load_target
from pentaceratops.evidence import evidence

prepared = RunResult.load("/path/to/cache/prepared/<candidate>_<hash>.npz")
target = load_target("/path/to/new_field")
results = evidence(prepared, target=target, likelihood="real", eb_eta=0.1)
```

`load_target` verifies the manifest and file hashes. It does not query catalogs,
read the original FITS, or rerun dilution. The same field can be used for Real
and Fourier evidence for the same candidate. Save a separate field for each
candidate: eligibility depends on the candidate's transit depth.

## Optional supplied inputs

Use an existing population to avoid submitting another TRILEGAL job:

```python
from pentaceratops import prepare_target

target = prepare_target(
    prepared, output_dir="/path/to/another_new_field",
    trilegal_fname="/path/to/existing_TRILEGAL.csv",
)
```

The existing population is copied byte for byte and must correspond to this
sky position. The current likelihood reader excludes the last two records;
adopted service CSVs with either one or two footer records are accepted with
that same historical row selection. They are not rewritten or reinterpreted.
New downloads include two explicit footer records so all downloaded stellar
rows reach the likelihood reader.

Other keyword arguments are:

| Argument | Meaning |
| --- | --- |
| `stars` | A pandas DataFrame or CSV containing adopted stellar inputs; bypasses TIC. Include `ID, Tmag, Jmag, Hmag, Kmag, ra, dec, mass, rad, Teff, plx`. Dilution is recomputed from the science FITS. |
| `transit_depth` | Optional measured **dimensionless aperture depth**, required if the saved bundle has no depth. For example, a 1000-ppm aperture depth is `0.001`. |
| `search_radius=10` | TIC cone radius in TESS pixels, using the existing 20.25-arcsecond convention. |
| `mag_lim=21` | Limiting magnitude sent to a new TRILEGAL query. |
| `population_timeout=900` | Maximum seconds polling the result after the TRILEGAL form query. |
| `poll_interval=10` | Seconds between result checks. |
| `trilegal_verify_ssl=True` | Verify HTTPS certificates for TRILEGAL query and download. Set `False` explicitly only to work around a certificate failure; this does not change TIC or global SSL settings. |

Supplied stellar inputs use solar mass/radius, kelvin, milliarcsecond parallax
and RA/Dec degrees. All sources need coordinates and TESS magnitudes for
aperture dilution. Eligible hosts require mass, radius and temperature.
J/H/K magnitudes and parallax are required for the target; resolved neighbors
may retain `NaN` because their NTP/NEB/NEBx2P fits do not use those fields.
No photometry is invented and no eligible neighbor is dropped. Missing
mass/radius/temperature values must be supplied from adopted stellar
information. The helper does not substitute solar values. To correct catalog
properties, supply a corrected `stars` table and write to a new field directory.
Record the sources of corrections and follow-up exclusions alongside it.

An optional prepared companion population remains an evidence argument,
`molusc_file=...`; it is separate from the TRILEGAL background population.

## INAF certificate failures

For `CERTIFICATE_VERIFY_FAILED`, an existing population supplied through
`trilegal_fname` avoids INAF entirely. For a new population, an explicit
request-local workaround is available:

```python
from pentaceratops import prepare_target

target = prepare_target(
    prepared, output_dir="/path/to/new_field",
    trilegal_verify_ssl=False,
)
```

This disables server-certificate authentication for both the TRILEGAL form
request and result download. It is opt-in, is recorded in the field provenance,
and does not change process-wide SSL settings or TIC catalog requests. The
default remains `True`; certificate failures fail immediately with a useful
message instead of waiting for the population timeout.

## Existing HZ fields

Reuse the adopted table and population to reproduce an existing HZ field,
including stellar corrections, dilution and follow-up exclusions. Do not
replace them with a new catalog query. Old fields without `field.json` can be
loaded directly for either TESS or Kepler:

```python
from types import SimpleNamespace
import pandas as pd

# prepared is the corresponding candidate bundle, already loaded above.
candidate = prepared.metadata["settings"]["candidate"]
target = SimpleNamespace(
    ID=int(candidate["host_id"]), mission=candidate["mission"],
    stars=pd.read_csv("/path/to/adopted_stars.csv"),
    trilegal_fname="/path/to/adopted_TRILEGAL.csv",
)
```

For older likelihood-ready folded files without candidate metadata, supply
the host ID and mission recorded by that run. `load_target` is specifically
for directories produced by `prepare_target`; the new TESS aperture recipe
must not be applied to Kepler FITS.
