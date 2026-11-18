# Examples

| File | Purpose |
| --- | --- |
| `tess_candidate.py` | Download and preprocess one TESS candidate |
| `kepler_candidate.py` | Download and preprocess one Kepler candidate |
| `prepare.slurm` | Slurm template for running one preparation as a job |
| `evidence.json` | Configuration template for `pentaceratops evidence` |

These scripts prepare a light curve; they do not calculate an FPP.

- [Candidate download and preparation](../docs/guide/candidate.md): how to run the
  scripts, catalogue sources, what preparation does, and where files are stored.
- [FPP from an existing or injected light curve](../docs/guide/supplied_lightcurve.md):
  starting from arrays that are already on disk.
