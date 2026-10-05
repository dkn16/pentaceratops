# Preparing the stellar-field target

`prepared` contains the light curve and its saved noise model. `target` contains
the host star, nearby sources, their aperture flux fractions and host-eligibility
depths, and the path to a TRILEGAL background-star population. The evidence call
needs both; `prepare_candidate` does not create the stellar field.

For an existing HZ run, reuse its saved stellar table and population, including
the adopted stellar parameters and any follow-up exclusions. A fresh catalog
query need not reproduce that field. For a new TESS candidate, the steps below
create the field files used in the [Python evidence examples](evidence_api.md).

## Create a new TESS field

Install the catalog extra with `python -m pip install -e '.[catalogs]'` from
the checkout. Catalog queries need network access. Start from the exact bundle
printed after `Prepared data:` by the preprocessing script, or use the
`prepared` object returned by `prepare_candidate`.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from pentaceratops import RunResult, Target

prepared = RunResult.load("/path/to/cache/prepared/<candidate>_<hash>.npz")
settings = prepared.metadata["settings"]
candidate = settings["candidate"]
if candidate["mission"] != "TESS":
    raise ValueError("This science-FITS aperture recipe is for TESS.")

field_dir = Path("/path/to/field").expanduser().resolve()  # One directory per candidate.
field_dir.mkdir(parents=True, exist_ok=True)
population_file = field_dir / "trilegal.csv"
target = Target(
    ID=int(candidate["host_id"]), mission="TESS",
    sectors=np.array([], dtype=int), search_radius=10,
    trilegal_fname=str(population_file),
)

# TIC may return IDs as strings; inference requires integer IDs and target first.
stars = target.stars.copy()
stars["ID"] = pd.to_numeric(stars["ID"], errors="raise").astype("int64")
is_host = stars["ID"] == target.ID
if is_host.sum() != 1:
    raise ValueError("The catalog must contain exactly one row for the target.")
target.stars = pd.concat([stars[is_host], stars[~is_host]], ignore_index=True)
```

`ID` here is the **TIC host ID**, obtained from the saved candidate metadata,
not the TOI number. The constructor queries TIC for the host and surrounding
stars. `sectors=[]` deliberately skips independent TessCut downloads: the next
step uses the FITS products that supplied the light curve. The explicit
population filename suppresses an automatic TRILEGAL request; an existing file
is reused, or the population step below creates it.

Check the queried stellar properties and apply any adopted corrections to
`target.stars` before inference. Mass/radius are in solar units, `Teff` in
kelvin, `plx` in milliarcseconds, and `ra`/`dec` in degrees. Record the sources
of corrections and any follow-up exclusions alongside the saved field.
The default evidence policy rejects missing required properties; it does not
infer a missing stellar mass or radius from the light-curve bundle.

## Calculate dilution using the actual aperture

Continue with the `prepared`, `settings`, and `target` above:

```python
from pentaceratops.preprocessing.aperture import geometry_from_fits, dilution_depths

geometries = [
    geometry_from_fits(
        product["source_file"], target.stars[["ra", "dec"]].to_numpy(),
        tic=target.ID, sector=int(product["sector"]),
    )
    for product in settings["native_products"]
]
depth = prepared.output["windows"].get("transit_depth")
if depth is None:
    raise ValueError("Supply a measured transit depth in the normalized aperture flux frame.")
fraction, host_depth, per_sector = dilution_depths(
    target.stars["Tmag"].to_numpy(), geometries, transit_depth=float(depth),
)
target.stars["fluxratio"] = fraction
target.stars["tdepth"] = host_depth
```

The original downloaded FITS files must remain available at their recorded
`source_file` paths. The helper uses each file's pipeline aperture and detector
WCS, with the same 0.75-pixel Gaussian PSF and equal sector weighting as
`Target.calc_depths`. See [aperture geometry](aperture-geometry.md).

`transit_depth` is a **dimensionless aperture depth**, not ppm. The preparation
bundle stores the catalog depth converted to that frame when a catalog depth
is available. Use that value without applying CROWDSAP again. If it is absent,
replace the `depth` assignment with a measured depth in the same aperture
frame. `tdepth` is the intrinsic depth each possible host would require;
sources requiring a depth greater than one become ineligible. This step is
needed before evidence: constructing `Target` alone does not populate
`fluxratio` or `tdepth`.

## Supply or create the background population

Reuse the adopted `trilegal.csv` for an existing field. If you do not have one,
this continues the setup above using the target's coordinates:

```python
from pentaceratops.stellar import query_TRILEGAL, save_trilegal

if not population_file.is_file():
    host = target.stars.iloc[0]
    # save_trilegal writes <ID>_TRILEGAL.csv in the current working directory.
    download_file = Path(f"{target.ID}_TRILEGAL.csv")
    if download_file.exists():
        raise FileExistsError(f"Move or reuse the existing population: {download_file}")
    url = query_TRILEGAL(float(host.ra), float(host.dec))
    if url is None:
        raise RuntimeError("TRILEGAL is unavailable; obtain the field population before inference.")
    downloaded = Path(save_trilegal(url, target.ID)).resolve()
    # Copy to the field directory, including when it is on another filesystem.
    with downloaded.open("rb") as source, population_file.open("xb") as destination:
        import shutil
        shutil.copyfileobj(source, destination)
target.trilegal_fname = str(population_file)
```

This is a separate online preparation step and can take time. A population
must correspond to this sky position and retain the format written by
`save_trilegal`; an empty placeholder is not a background model. The evidence
function requires the local population file and does not download it. An
optional prepared companion population can be supplied separately through
`molusc_file=...`.

## Save, reload, and run evidence

The object is now ready for `evidence(prepared, target=target, ...)`. Save the
table for subsequent offline runs:

```python
target.stars.to_csv(field_dir / "stars.csv", index=False, mode="x")
```

Later, the full call using the two saved inputs is:

```python
from types import SimpleNamespace
import pandas as pd
from pentaceratops import RunResult
from pentaceratops.evidence import evidence

prepared = RunResult.load("/path/to/cache/prepared/<candidate>_<hash>.npz")
candidate = prepared.metadata["settings"]["candidate"]
target = SimpleNamespace(
    ID=int(candidate["host_id"]), mission=candidate["mission"],
    stars=pd.read_csv("/path/to/field/stars.csv"),
    trilegal_fname="/path/to/field/trilegal.csv",
)
results = evidence(prepared, target=target, likelihood="fourier", N=500, steps=50, eb_eta=0.1)
```

`SimpleNamespace` is sufficient because inference reads these attributes; it
does not need the constructor's catalog-query or plotting methods. Set
`likelihood="real"` for Real evidence with the same field. This reload route
also applies to the saved Kepler fields: preserve their existing host ID,
mission, table, population, and dilution. The new-field aperture recipe above
is specifically for TESS FITS and must not be applied to Kepler products.
