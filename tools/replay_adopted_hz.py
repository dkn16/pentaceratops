#!/usr/bin/env python3
"""Read-only replay of adopted folded HZ fits and their stored evidences.

Requires the archived campaigns and scratch products. Does not run a sampler,
download data, modify the paper, or regenerate the large covariance matrices.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pentaceratops import FoldedFourierData, load_folded_real, scenario_probabilities
from pentaceratops.preprocessing.joint_fourier import load_block


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probabilities(frame, expected, expected_eb):
    table = scenario_probabilities(frame, evidence_column="lnBF", eb_eta=.1)
    np.testing.assert_allclose(table.attrs["FPP"], expected, atol=2e-12, rtol=0)
    np.testing.assert_allclose(table.attrs["FPP_EB"], expected_eb, atol=2e-12, rtol=0)
    return {key: table.attrs[key] for key in ("FPP", "FPP_EB", "NFPP")}


def tess(research, scratch):
    campaign = research/"tess_test/tess_habitable_zone/hz_fourier_folded_7390_20261002"
    bulk = scratch/"tess_test/hz_fourier_folded_7390_20261002"
    block = load_block(bulk/"folded_block")
    with np.load(bulk/"fold_map.npz", allow_pickle=False) as z:
        folded = FoldedFourierData(block, z["native_time"], z["bin_index"], z["weights"],
                                   float(z["period"]), float(z["epoch"]))
    adapter = folded.adapter(mission="TESS", nsamples=7, parity="profile")
    comparisons = []
    for directory in sorted((bulk/"full/TIC160075684_TOI7390.01").iterdir()):
        best = json.loads((directory/"bestfit.json").read_text())
        adapter.aperture_fraction = best["aperture_fraction"]
        model = adapter.native_model(best["kind"], best["parameters"],
                                     reverse=best.get("reversed_parity", False))
        with np.load(directory/"samples.npz", allow_pickle=False) as z:
            error = float(np.max(np.abs(model-z["bestfit_model"])))
            np.testing.assert_allclose(model, z["bestfit_model"], atol=2e-12, rtol=0)
        gain = float(adapter.metric.gain(model))
        np.testing.assert_allclose(gain, best["photometric_loglike_ratio"], atol=2e-5, rtol=2e-12)
        comparisons.append(dict(scenario=directory.name, model_max_error=error,
                                gain_error=gain-best["photometric_loglike_ratio"]))
    assert len(comparisons) == 21
    return dict(scenarios=len(comparisons), native_samples=len(folded.native_time),
                folded_bins=len(block.time), replay=comparisons,
                **probabilities(pd.read_csv(campaign/"full_scenarios.csv"),
                                .11453120078512385, .09340197090185902))


def kepler(research, scratch):
    from types import SimpleNamespace
    from pentaceratops.experimental.folded_baseline import FoldedBaselineObservation
    from pentaceratops.likelihoods.joint_fourier import JointFourierMetric
    from pentaceratops.likelihoods import real as lk

    campaign = research/"kepler_injection/FourierLikelihood/koi2719_folded_subsets_20261002"
    bulk = scratch/"koi2719_folded_subsets_20261002"
    baseline = module_at("hz_original_callback_capture", campaign/"baseline.py")
    baseline.OLD = research/"kepler_injection/FourierLikelihood/hz_fourier_completion_20261001"
    baseline.OLD_BULK = scratch/"kepler_hz_fourier_completion_20261001"
    row = next(r for r in json.loads((campaign/"selected_manifest.json").read_text())
               if r["tag"] == "7_transits")
    adapters = {}
    for representation in ("P", "2P"):
        directory = bulk/"blocks/7_transits"/representation
        meta = json.loads((directory/"fold.json").read_text())
        with np.load(directory/"selection.npz", allow_pickle=False) as z:
            ids = z["indices"].copy()
        adapters[representation] = SimpleNamespace(
            metric=JointFourierMetric([load_block(directory)]), period=row["period"],
            epoch=row["epoch_btjd"], exptime=row["cadence"], nsamples=7, lk=lk,
            record=True, snapshot=None, aperture_fraction=1., fold_indices=ids,
            fold_length=meta["fold_length"], fold_roll=meta["fold_roll"],
        )
    comparisons = []
    for task in baseline.plan():
        directory = bulk/"full/7_transits"/(str(task["ID"])+"_"+task["scenario"])
        best = json.loads((directory/"bestfit.json").read_text())
        adapter = adapters["2P" if task["scenario"].endswith("x2P") else "P"]
        adapter.aperture_fraction = task["star"]["fluxratio"]
        captured = baseline.capture(task)
        bridge = FoldedBaselineObservation(captured["observation"], adapter)
        with np.load(directory/"samples.npz", allow_pickle=False) as z:
            score = float(bridge(z["bestfit_theta"]))
            error = float(np.max(np.abs(adapter.snapshot["model"]-z["bestfit_model"])))
            np.testing.assert_allclose(adapter.snapshot["model"], z["bestfit_model"], atol=2e-12, rtol=0)
        np.testing.assert_allclose(score, best["log_target"], atol=2e-5, rtol=2e-12)
        comparisons.append(dict(scenario=directory.name, model_max_error=error,
                                log_target_error=score-best["log_target"]))
    assert len(comparisons) == 18
    frame = pd.read_csv(campaign/"full_scenarios.csv")
    return dict(scenarios=len(comparisons), events=row["events"], replay=comparisons,
                **probabilities(frame.loc[frame.tag == "7_transits"],
                                .029638123704248648, .002594648113663618))


def real_tess(research):
    from pentaceratops.hz import real_adapter

    paper = research/"papers/Pentaceratops/results/tess_hz_20261001"
    provenance = json.loads((paper/"provenance.json").read_text())
    comparisons = []
    targets = []
    for source in provenance["table_sources"]:
        summary_path = Path(source["source_records"]["Real"]["source"])
        summary = json.loads(summary_path.read_text())
        campaign = summary_path.parents[3]
        manifest = json.loads((campaign/"prepared_manifest.json").read_text())
        row = next(r for r in manifest if r["candidate"] == source["candidate"])
        data = load_folded_real(row["v2_prepared_path"], row["v2_folded_path"])
        adapter = real_adapter(data, mission="TESS", nsamples=20, parity="profile", timing_policy="legacy")
        adapter.record = True
        for directory in sorted((summary_path.parent/"scenarios").iterdir()):
            path = directory/"bestfit_parameters.json"
            if not path.exists():
                result = json.loads((directory/"result.json").read_text())
                assert result["lnZ"] in (None, -np.inf)
                continue
            best = json.loads(path.read_text())
            adapter.aperture_fraction = best["aperture_fraction"]
            method = getattr(adapter, best["kind"])
            placeholders = [None] * (3 if best["kind"] == "planet" else 6)
            gain = -float(method(*placeholders, **best["parameters"]))
            np.testing.assert_allclose(gain, best["photometric_loglike_ratio"], atol=2e-5, rtol=2e-12)
            comparisons.append(dict(candidate=source["candidate"], scenario=directory.name,
                                    gain_error=gain-best["photometric_loglike_ratio"]))
        frame = pd.read_csv(summary_path.parent/"scenarios.csv").rename(columns={"lnZ": "lnBF"})
        targets.append(dict(candidate=source["candidate"],
                            **probabilities(frame, summary["FPP_eta0p1"], summary["FPP_EB_eta0p1"])))
    assert len(targets) == 7
    return dict(targets=targets, scenarios=len(comparisons), replay=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(tess7390=tess(args.research_root, args.scratch_root),
                  kepler2719=kepler(args.research_root, args.scratch_root),
                  real_tess=real_tess(args.research_root),
                  scope="Saved best-fit models/log targets and stored evidence normalization; no resampling")
    from pentaceratops.api import _source_fingerprint
    report["source_sha256"] = _source_fingerprint()
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({name: {k: v for k, v in record.items() if k != "replay"}
                      for name, record in report.items() if isinstance(record, dict)}, indent=2))


if __name__ == "__main__":
    main()
