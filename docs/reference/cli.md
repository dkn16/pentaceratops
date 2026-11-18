# Command line

The package installs one command, `pentaceratops`, which can also be run as
`python -m pentaceratops`. Two example scripts in the repository cover
light-curve preparation.

## `pentaceratops`

```text
pentaceratops [--version] {doctor,sampling-policy,inspect,evidence} ...
```

| Subcommand | Purpose |
| --- | --- |
| `doctor` | Print the installed version of each dependency. Exits non-zero if a required one is missing. No queries or downloads. |
| `sampling-policy` | Print the default `N` and `steps` of every scenario as JSON. |
| `inspect PATH` | Summarize a saved result file: package version, function, number of sampler calls and stored particles. |
| `evidence CONFIG` | Run the evidence calculation from prepared files described by a JSON configuration. |

### `pentaceratops evidence`

```bash
pentaceratops evidence target.json
pentaceratops evidence target.json --likelihood fourier --output results/fourier.npz
```

| Flag | Meaning |
| --- | --- |
| `--likelihood {real,fourier}` | Override the likelihood named in the configuration. |
| `--output PATH` | New result file, relative to the current directory. |
| `--N`, `--steps` | Sampling effort for every scenario. |
| `--nsamples` | Exposure-integration points. |
| `--seed` | Random seed. |
| `--posterior-samples` | Posterior draws kept per scenario. |
| `--eb-eta` | Weight of the eclipsing-binary scenarios. |

Flags override the configuration file. Settings without a flag, such as
`free_epoch`, `epoch_sigma` and `scenarios`, go in the `settings` block of the
JSON. The command prints the output path, the number of scenarios, and `FPP`,
`FPP_EB` and `NFPP`.

The configuration format, the stellar-field files it refers to, and what the
command does with the saved photometry are described in
[run from the command line](../evidence_command.md).

## Preparation scripts

`examples/tess_candidate.py` and `examples/kepler_candidate.py` download and
preprocess one candidate. They do not compute an FPP; they print the path of
the prepared file to pass to `evidence`.

```bash
python examples/tess_candidate.py --target TOI-4616.01
python examples/kepler_candidate.py --target 'KIC 8758204'
```

| Flag | Meaning |
| --- | --- |
| `--target` | TIC/TOI or KIC/KOI identifier. |
| `--candidate` | TOI or KOI to use when the host has several. |
| `--list-candidates` | List the catalogued signals of the host and exit. |
| `--stage {resolve,download,prepare}` | Stop after resolving the ephemeris, after downloading, or run the full preparation (default). |
| `--catalog-source {nasa,bayesian}` | Ephemeris catalogue. `bayesian` is available for TESS. |
| `--catalog-file` | Local copy of the Bayesian TESS catalogue. |
| `--sectors`, `--quarters`, `--segments` | Restrict to these sectors or quarters. |
| `--cadence-seconds` | Select a specific archive cadence. |
| `--period`, `--epoch-bjd`, `--duration-hours`, `--depth-ppm` | Override the catalogue ephemeris. The epoch is full BJD. |
| `--mask-planet PERIOD,BJD,DURATION_HOURS` | Mask an additional signal; repeatable. |
| `--keep-other-planets` | Do not mask other catalogued planets (diagnostic only). |
| `--cache-dir` | Cache directory. |
| `--plot-dir`, `--no-plots` | Where to write diagnostic plots, or skip them. |
| `--plot-only PREPARED_NPZ` | Redraw the plots of a saved preparation without any network access or fitting. |
| `--offline` | Use only existing caches. |

The preprocessing settings (`--bin-minutes`, `--window-durations` and the
others) are listed under [options](options.md#light-curve-preparation). Run
either script with `--help` for the complete list; this makes no catalogue
query.

`examples/prepare.slurm` is a template for running one preparation as a Slurm
job.
