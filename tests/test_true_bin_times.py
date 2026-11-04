"""Regridded bins keep the samples' true mean time, and the windowed likelihood folds the model as the data."""
import numpy as np
import pytest

from pentaceratops.preprocessing.candidate import regular_bin, sample_times, phase
from pentaceratops.preprocessing.evidence_inputs import candidate_folded_real
from pentaceratops.hz import real_adapter
from test_prepared_command import prepared
from test_exposure import physical_columns, scalar_scores, assert_agreement


def test_regular_bin_returns_true_mean_time():
    dt = 30 / 1440
    offset = 7 / 1440                                  # samples sit 7 min after every grid node
    time = 100 + np.arange(50) * dt + offset
    grid, flux, error, true_time = regular_bin(time, np.ones(50), np.full(50, 1e-3), dt)
    good = np.isfinite(flux)
    np.testing.assert_allclose(true_time[good], time, atol=1e-10)
    np.testing.assert_allclose((true_time - grid)[good], offset, atol=1e-10)   # the grid alone would be off by 7 min


def test_regular_bin_true_time_is_mean_of_bin_samples():
    dt = 30 / 1440
    time = 100 + np.arange(30) * (10 / 1440) + 2 / 1440   # 10-min samples averaged into 30-min bins
    grid, flux, error, true_time = regular_bin(time, np.ones(30), np.full(30, 1e-3), dt)
    ids = np.floor((time - grid[0]) / dt + .5).astype(int)
    for i in np.unique(ids):
        np.testing.assert_allclose(true_time[i], time[ids == i].mean(), atol=1e-10)


def test_sample_times_prefers_true_time_and_falls_back_to_the_grid():
    grid = np.array([1., 2.])
    np.testing.assert_array_equal(sample_times(dict(time=grid)), grid)                              # old bundles
    np.testing.assert_allclose(sample_times(dict(time=grid, true_time=np.array([1.1, np.nan]))), [1.1, 2.])


def test_folded_real_native_map_reproduces_the_data_fold(monkeypatch):
    saved, ephem, _ = prepared(monkeypatch, two_sectors=True)
    data = candidate_folded_real(saved)
    dt = data["exptime"]
    for i, panel in enumerate(("even", "odd", "secondary")):
        time, index, weight = data[panel+"_native_time"], data[panel+"_bin_index"], data[panel+"_weight"]
        centers = data[panel+"_time"]
        assert index.dtype.kind == "i" and len(time) == len(index) == len(weight) > len(centers)
        np.testing.assert_allclose(np.bincount(index, weights=weight, minlength=len(centers)), 1., atol=1e-12)
        # every retained grid bin lies inside its folded bin, judged by its true phase
        assert np.all(np.abs(time - centers[index]) <= dt / 2 + 1e-12)
        # folding the sectors' residual flux with the map reproduces the saved folded flux exactly
        native_flux = []
        for sector in saved.output["segments"]:
            t = sample_times(sector)
            reference = ephem.epoch_relative + (ephem.period_days / 2 if panel == "secondary" else 0.)
            rel, cycles = phase(t, reference, ephem.period_days)
            inside = sector["observed"] & (np.abs(rel) <= centers[-1] + dt / 2)
            if panel != "secondary":
                inside &= cycles % 2 == int(panel == "odd")
            native_flux.append(np.asarray(sector["residual_flux"])[inside])
        folded = np.bincount(index, weights=weight * np.concatenate(native_flux), minlength=len(centers))
        np.testing.assert_allclose(folded, data[panel+"_map_flux"], atol=1e-12, rtol=0)


def test_windowed_adapter_folds_the_model_as_the_data(monkeypatch):
    saved, _, _ = prepared(monkeypatch)
    data = candidate_folded_real(saved)
    adapter = real_adapter(data, mission="TESS", nsamples=4, timing_policy="legacy")
    assert adapter.model_times == "native_fold"
    # batched kernel == explicit fold of the scalar simulator at the native phases
    columns = physical_columns("planet")
    columns["P_orb"] = np.full(len(columns["P_orb"]), data["period"])
    batched = adapter.dilute(adapter.undiluted(columns, "primary", np.zeros(len(columns["P_orb"])), planet=True), columns, planet=True)
    for j in range(len(columns["P_orb"])):
        pars = {k: v[j] for k, v in columns.items() if k not in ("R_EB", "EB_fluxratio")}
        native = adapter.lk.simulate_TP_transit(adapter.model_time["primary"], exptime=data["exptime"], nsamples=4, **pars)
        np.testing.assert_allclose(batched[j], adapter.fold_model("primary", native), atol=2e-9, rtol=0)
    # scalar (replay) path and batched path agree for every scenario kind
    for kind in ("planet", "binary", "x2p"):
        columns = physical_columns(kind)
        columns["P_orb"] = np.full(len(columns["P_orb"]), data["period"] * (2 if kind == "x2p" else 1))
        reference = scalar_scores(adapter, kind, columns, nsamples=4)
        assert_agreement(reference, adapter.photometric_columns(kind, columns, nsamples=4))


def test_windowed_adapter_without_a_map_uses_bin_centres(monkeypatch):
    saved, _, _ = prepared(monkeypatch)
    data = candidate_folded_real(saved)
    old = {k: v for k, v in data.items() if not k.endswith(("_native_time", "_bin_index", "_weight"))}
    adapter = real_adapter(old, mission="TESS")
    assert adapter.model_times == "bin_centers" and adapter.fold_operator is None
    np.testing.assert_array_equal(adapter.model_time["primary"], adapter.sorted_eo_time)
    partial = dict(old, even_native_time=data["even_native_time"])
    with pytest.raises(ValueError, match="Incomplete native fold map"):
        real_adapter(partial, mission="TESS")
