"""Offline tests: catalog conventions, downloads, protection/folding, and caches."""
from dataclasses import asdict
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pentaceratops.preprocessing import catalog, candidate as prep, download
from pentaceratops.results import RunResult


@pytest.mark.parametrize("text,mission,kind,value", [
    ("TIC 123", "TESS", "TIC", "123"), ("123", "Kepler", "KIC", "123"),
    ("TOI-700.02", "TESS", "TOI", "700.02"), ("KOI-72.1", "Kepler", "KOI", "72.10"),
    ("K00072.01", "Kepler", "KOI", "72.01"),
])
def test_identifiers(text, mission, kind, value):
    result = catalog.parse_identifier(text, mission)
    assert (result.mission, result.kind, result.value) == (mission, kind, value)


@pytest.mark.parametrize("text", ["TOI700", "TIC0", "KIC-1.02", "KOI72.001", "TIC 1 or 1=1", "WASP-1"])
def test_unsafe_or_incomplete_identifiers_fail(text):
    with pytest.raises(ValueError):
        catalog.parse_identifier(text)


def tess_row(toi="1.01", **kwargs):
    return dict(tid=123, toi=toi, pl_orbper=2., pl_tranmid=2459000., pl_trandurh=2.,
                pl_trandep=1000., tfopwg_disp="PC", **kwargs)


def test_nasa_default_ambiguity_selection_companions_and_override(monkeypatch, tmp_path):
    rows = [tess_row(), tess_row("1.02")]
    def fetch(ident, *args, **kwargs):
        assert kwargs["source"] == "nasa"
        return rows, {"query": "fixture"}
    monkeypatch.setattr(catalog, "catalog_rows", fetch)
    with pytest.raises(ValueError, match="multiple candidates"):
        catalog.resolve_candidate("TIC 123", tmp_path)
    selected, companions, metadata = catalog.resolve_candidate("TIC 123", tmp_path,
        candidate="TOI1.02", overrides={"period_days": 3.})
    assert selected.period_days == 3. and selected.epoch_relative == 2000.
    assert [c.identifier for c in companions] == ["TOI 1.01"]
    assert metadata["source"] == "nasa"
    with pytest.raises(ValueError, match="not uniquely associated"):
        catalog.resolve_candidate("TIC123", tmp_path, candidate="TOI2.01")


def test_kepler_epoch_conversion_and_missing_period():
    row = dict(kepoi_name="K00072.01", koi_period=.8, koi_time0bk=123., koi_duration=2.)
    selected = catalog.candidate_from_row(row, "Kepler", 42)
    assert selected.epoch_bjd == 2454956. and selected.epoch_relative == 123.
    with pytest.raises(ValueError, match="full BJD"):
        catalog.candidate_from_row(row, "Kepler", 42, {"epoch_bjd": 123.})
    with pytest.raises(ValueError, match="Missing ephemeris"):
        catalog.candidate_from_row({}, "TESS", 42, identifier="custom")


def test_incomplete_companion_ephemeris_is_not_silently_ignored(monkeypatch, tmp_path):
    rows = [tess_row(), dict(tess_row("1.02"), pl_orbper=None)]
    monkeypatch.setattr(catalog, "catalog_rows", lambda *a, **k: (rows, {}))
    with pytest.raises(ValueError, match="Missing ephemeris"):
        catalog.resolve_candidate("TIC123", tmp_path, candidate="TOI1.01")


def test_bayesian_epoch_duration_not_phase_tau_and_no_nasa_fallback(monkeypatch, tmp_path):
    row = dict(TIC="123", TOI="1.01", Period="2.1", Epoch="2100.5", Duration="0.1",
               Phase="0.3", Tau="0.05", passed_all_tests="True")
    monkeypatch.setattr(catalog, "bayesian_rows", lambda *a, **k: ([row], {"source": "bayesian"}))
    monkeypatch.setattr(catalog, "archive_query", lambda *a, **k: pytest.fail("No NASA fallback"))
    chosen, _, meta = catalog.resolve_candidate("TOI1.01", tmp_path, source="bayesian")
    assert chosen.epoch_bjd == 2459100.5
    assert chosen.duration_hours == pytest.approx(2.4)
    assert chosen.depth_ppm is None and meta["source"] == "bayesian"
    with pytest.raises(ValueError, match="Expected one"):
        catalog.resolve_candidate("TOI99.01", tmp_path, source="bayesian")
    with pytest.raises(ValueError, match="TESS-only"):
        catalog.resolve_candidate("KOI1.01", tmp_path, source="bayesian")
    row["Epoch"] = "2459100.5"
    with pytest.raises(ValueError, match="must be BTJD"):
        catalog.resolve_candidate("TOI1.01", tmp_path, source="bayesian")


def test_bayesian_pinned_local_schema(tmp_path):
    # Runtime test fixtures, not catalog downloads.
    path = tmp_path / "catalog.csv"
    path.write_text("TIC,TOI,Period,Epoch,Duration\n123,1.01,2,2000,.1\n")
    a, meta_a = catalog.bayesian_rows(tmp_path, catalog_file=path)
    b, meta_b = catalog.bayesian_rows(tmp_path, catalog_file=path)
    assert a == b and meta_a == meta_b
    path.write_text("TIC,TOI,Period,Phase,Tau\n123,1.01,2,.3,.1\n")
    with pytest.raises(ValueError, match="absolute ephemeris"):
        catalog.bayesian_rows(tmp_path, catalog_file=path)


def test_archive_pin_reuse_without_network(monkeypatch, tmp_path):
    import io
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        return io.BytesIO(b'[{"tid":123}]')
    monkeypatch.setattr(catalog, "urlopen", fetch)
    rows, metadata = catalog.archive_query("select tid from toi where tid=123", tmp_path)
    other, provenance = catalog.archive_query("select tid from toi where tid=123", tmp_path, offline=True)
    assert rows == other and metadata == provenance and len(calls) == 1
    with pytest.raises(FileNotFoundError):
        catalog.archive_query("not cached", tmp_path, offline=True)


def test_product_selection_before_download():
    rows = [dict(mission="TESS Sector 17", author=author, exptime=exposure, productFilename=f"{i}.fits")
            for i, (author, exposure) in enumerate([( "TESS-SPOC", 1800), ("SPOC", 20), ("SPOC", 120)])]
    rows.append(dict(mission="TESS Sector 18", author="SPOC", exptime=120, productFilename="4.fits"))
    assert download.product_selection(rows, "TESS", [17]) == [(17, 2)]
    assert download.product_selection(rows, "TESS", [17], 20) == [(17, 1)]
    with pytest.raises(ValueError, match="requested segments"):
        download.product_selection(rows, "TESS", [19])


@pytest.mark.parametrize("mission", ["TESS", "Kepler"])
def test_product_time_frame_dilution_and_target_identity(mission):
    from astropy.time import Time
    ephem = catalog.Candidate(mission, 123, "test", 2., 2459000., 2.)
    meta = {"TICID" if mission == "TESS" else "KEPLERID": 123,
            "SECTOR" if mission == "TESS" else "QUARTER": 17,
            "FLUX_ORIGIN": "pdcsap_flux", "CROWDSAP": .4}
    lc = SimpleNamespace(time=Time([2459000., 2459000.01, 2459000.02], format="jd", scale="tdb"),
                         flux=SimpleNamespace(value=np.array([100., 98., 100.])),
                         flux_err=SimpleNamespace(value=np.ones(3)), meta=meta)
    data, provenance = download.normalize_product(lc, ephem, 17, 120)
    assert data["time"][0] == 2459000. - ephem.time_zero
    np.testing.assert_allclose(data["flux"], [1., .992, 1.])
    np.testing.assert_allclose(data["error"], .004)
    assert provenance["crowding_restore_factor"] == .4
    with pytest.raises(ValueError, match="sector/quarter"):
        download.normalize_product(lc, ephem, 18, 120)
    meta.pop("CROWDSAP")
    with pytest.raises(ValueError, match="CROWDSAP"):
        download.normalize_product(lc, ephem, 17, 120)


def fixtures():
    ephem = catalog.Candidate("TESS", 123, "TOI 1.01", 2.31, 2457001.07, 2.)
    t = np.arange(0, 14, .01)
    rel, _ = prep.phase(t, ephem.epoch_relative, ephem.period_days)
    rng = np.random.default_rng(14)
    flux = 1 + .0003*np.sin(t*2) - .002*(abs(rel)<ephem.duration_days/2) + rng.normal(0, .0001, len(t))
    block = dict(time=t, flux=flux, error=np.full(len(t), .0002), segment=1, exposure_seconds=864.)
    options = prep.PreparationOptions(bin_minutes=14.4, window_durations=2., padding_fraction=.1)
    return ephem, block, options


def fake_fgp(time, flux, error, protected, dt, **kwargs):
    observed = np.isfinite(flux)
    model = np.full(len(time), .0001)
    return dict(cleaned_flux=flux.copy(), model_flux=model, residual_flux=flux-model,
                relative_error=error, observed=observed, protected=protected,
                fit_mask=observed & ~protected, center=1., scale=.001, mode_count=1,
                fit_values=np.nan_to_num(flux-1)/.001, stellar_psd=np.ones(len(time)//2),
                white_variance=.01, psd_total=np.ones(len(time)//2))


def test_protection_matches_every_used_point_and_gaps_never_become_data(monkeypatch):
    ephem, block, options = fixtures()
    block["flux"][15:20] = np.nan
    monkeypatch.setattr(prep, "detrend_regular_segment", fake_fgp)
    result, audit = prep.preprocess_blocks([block], ephem, options=options)
    stream, windows = result["stream"], result["windows"]
    assigned, science = prep.assignments(stream["time"], np.ones(len(stream["time"]), bool),
                                         ephem, windows["time_even"], windows["bin_days"])
    np.testing.assert_array_equal(science, stream["protected"])
    segment = result["segments"][0]
    assert not np.any(segment["fit_mask"] & segment["protected"])
    assert len(stream["time"]) == len(block["time"])-5
    assert sum(np.sum(use) for _, use in assigned.values()) == int(science.sum())
    assert audit[0]["training_likelihood_overlap"] == 0
    assert np.nanmin(windows["flux_primary"]) < .999
    # Combined primary diagnostic equals the inverse-variance parity combination.
    we, wo = 1/windows["err_even"]**2, 1/windows["err_odd"]**2
    np.testing.assert_allclose(windows["flux_primary"],
                              (we*windows["flux_even"]+wo*windows["flux_odd"])/(we+wo))


def test_full_preparation_cache_and_plot_replay(monkeypatch, tmp_path):
    ephem, block, options = fixtures()
    monkeypatch.setattr(prep, "resolve_candidate", lambda *a, **k: (ephem, [], {"queries": [], "source": "nasa"}))
    monkeypatch.setattr(prep, "download_products", lambda *a, **k: ([block], {"selection": {}, "products": []}, tmp_path/"native.npz"))
    monkeypatch.setattr(prep, "detrend_regular_segment", fake_fgp)
    result, path = prep.prepare_candidate("TIC123", tmp_path, options=options)
    monkeypatch.setattr(prep, "preprocess_blocks", lambda *a, **k: pytest.fail("Do not refit cached preparation"))
    again, same = prep.prepare_candidate("TIC123", tmp_path, options=options, offline=True)
    assert path == same and again.metadata["likelihood_ready"] is False
    np.testing.assert_array_equal(result.output["windows"]["flux_primary"], again.output["windows"]["flux_primary"])
    paths = prep.plot_preparation(RunResult.load(path), tmp_path / "plots")
    assert len(paths) == 2 and all(p.stat().st_size > 1000 for p in paths)


def test_companion_mask_removes_signal_before_binning(monkeypatch):
    ephem, block, options = fixtures()
    companion = catalog.Candidate("TESS", 123, "manual", 3.7, 2457000.4, 1.)
    rel, _ = prep.phase(block["time"], companion.epoch_relative, companion.period_days)
    block["flux"][abs(rel) < companion.duration_days/2] -= .1
    monkeypatch.setattr(prep, "detrend_regular_segment", fake_fgp)
    result, audit = prep.preprocess_blocks([block], ephem, [companion], options)
    assert audit[0]["masked_native_samples"] > 0
    assert np.nanmin(result["segments"][0]["raw_flux"]) > .99


def test_insufficient_training_or_parity_is_an_explicit_failure(monkeypatch):
    ephem, block, options = fixtures()
    short = {key: value[:150] if isinstance(value, np.ndarray) else value for key, value in block.items()}
    with pytest.raises(ValueError, match="training bins"):
        prep.preprocess_blocks([short], ephem, options=options)
    monkeypatch.setattr(prep, "detrend_regular_segment", fake_fgp)
    _, cycles = prep.phase(block["time"], ephem.epoch_relative, ephem.period_days)
    block["flux"][cycles % 2 == 1] = np.nan
    with pytest.raises(ValueError, match="Incomplete odd window"):
        prep.preprocess_blocks([block], ephem, options=options)


def test_returned_fgp_state_replays_last_map_objective(monkeypatch):
    from pentaceratops.preprocessing import fgp
    monkeypatch.setattr(fgp, "initial_gap_model", lambda t, y, mask: np.zeros(len(t)))
    t = np.arange(512) * .03
    f = 1 + .001*np.sin(t) + np.random.default_rng(2).normal(0, .0001, len(t))
    mask = np.zeros(len(t), bool)
    mask[50:70] = True
    output = fgp.detrend_regular_segment(t, f, np.full(len(t), .0001), mask, .03, iterations=1, cutoff_per_day=.5)
    replay, count = fgp.fit_fourier_modes(output["fit_values"], output["fit_mask"], output["stellar_psd"],
                                         output["white_variance"], .03, .5)
    np.testing.assert_allclose(output["model_flux"], output["scale"]/output["center"]*replay)
    assert count == output["mode_count"]


def test_without_celerite_initializer_is_not_exact_training_interpolation(monkeypatch):
    import builtins
    from pentaceratops.preprocessing import fgp
    original_import = builtins.__import__
    def without_celerite(name, *args, **kwargs):
        if name == "celerite":
            raise ImportError("test missing optional celerite")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_celerite)
    t = np.arange(512) * .03
    y = np.sin(t) + np.random.default_rng(22).normal(0, .1, len(t))
    train = np.ones(len(t), bool)
    train[50:70] = False
    with pytest.warns(UserWarning, match="smoothed interpolation"):
        initial = fgp.initial_gap_model(t, y, train)
    assert np.std((y-initial)[train]) > 0
    changed = y.copy()
    changed[~train] = 1e9
    with pytest.warns(UserWarning, match="smoothed interpolation"):
        np.testing.assert_array_equal(initial, fgp.initial_gap_model(t, changed, train))
    with pytest.warns(UserWarning, match="smoothed interpolation"):
        result = fgp.detrend_regular_segment(t, 1 + .001*y, np.full(len(t), .0001),
                                            ~train, .03, iterations=1, cutoff_per_day=.5)
    assert np.all(np.isfinite(result["residual_flux"]))


@pytest.mark.parametrize("mission,default", [("TESS", "TOI-4616.01"), ("Kepler", "KIC8758204")])
def test_help_is_offline(mission, default):
    code = f"from pentaceratops.preprocessing.example_cli import main; main({mission!r}, {default!r}, ['--help'])"
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    assert "--catalog-source" in result.stdout and "--cache-dir" in result.stdout
