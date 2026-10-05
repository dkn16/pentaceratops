"""Stellar-field setup: real FITS geometry, mocked network, and evidence inputs."""
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from pentaceratops import prepare_target, load_target
from pentaceratops.evidence import evidence
from pentaceratops.preprocessing import field
from pentaceratops.stellar import trilegal_results
from test_aperture import product
from test_prepared_command import prepared as candidate_fixture


RAW_POPULATION = b'''#Gc Mact logg logTe [M/H] TESS J H Ks
1 1.0 4.4 3.76 0.0 14.0 13.5 13.2 13.0
1 0.8 4.5 3.65 -0.1 16.0 15.5 15.2 15.0
# total objects
#TRILEGAL normally terminated
'''


def population_file(path):
    rows = pd.DataFrame({"Mact": [1., .8, np.nan, np.nan], "logg": [4.4, 4.5, np.nan, np.nan],
        "logTe": [3.76, 3.65, np.nan, np.nan], "[M/H]": [0., -.1, np.nan, np.nan],
        "TESS": [14., 16., np.nan, np.nan], "J": [13.5, 15.5, np.nan, np.nan],
        "H": [13.2, 15.2, np.nan, np.nan], "Ks": [13., 15., np.nan, np.nan],
        "record": ["star", "star", "# total objects", "#TRILEGAL normally terminated"]})
    rows.to_csv(path, index=False)
    return path


@pytest.fixture
def inputs(monkeypatch, tmp_path):
    saved, _, _ = candidate_fixture(monkeypatch)
    filename, wcs = product(tmp_path)
    with fits.open(filename, mode="update") as hdus:
        hdus[0].header["TICID"] = 123
    sky = wcs.all_pix2world([[3., 2.], [4.1, 2.7]], 0)
    rows = []
    for index, (ident, mag) in enumerate(((123, 10.), (456, 12.))):
        rows.append(dict(ID=str(ident), Tmag=mag, Jmag=mag-.3, Hmag=mag-.4, Kmag=mag-.5,
            ra=sky[index, 0], dec=sky[index, 1], mass=1., rad=1., Teff=5777., plx=5.))
    stars = pd.DataFrame(rows[::-1])
    saved.metadata["settings"]["native_products"] = [dict(source_file=str(filename), sector=40,
        fits_sha256=hashlib.sha256(filename.read_bytes()).hexdigest())]
    saved.output["windows"]["transit_depth"] = .01
    population = population_file(tmp_path/"input_population.csv")
    return saved, stars, population


def no_network(monkeypatch):
    monkeypatch.setattr(field, "_query_stars", lambda *a: pytest.fail("No catalog query"))
    monkeypatch.setattr(field, "_download_population", lambda *a, **k: pytest.fail("No population query"))


def test_supplied_inputs_publish_field_and_reload_offline(monkeypatch, tmp_path, inputs):
    saved, stars, population = inputs
    original = stars.copy(deep=True)
    no_network(monkeypatch)
    bundle = saved.save(tmp_path/"candidate.npz")
    destination = tmp_path/"field"
    result = prepare_target(bundle, destination, stars=stars, trilegal_fname=population)
    assert result.ID == 123 and result.mission == "TESS"
    assert result.stars.ID.tolist() == [123, 456]
    np.testing.assert_allclose(result.stars.fluxratio.sum(), 1.)
    np.testing.assert_allclose(result.stars.tdepth*result.stars.fluxratio, .01)
    assert (destination/"trilegal.csv").read_bytes() == population.read_bytes()
    assert set(p.name for p in destination.iterdir()) == {"stars.csv", "trilegal.csv", "field.json"}
    pd.testing.assert_frame_equal(load_target(destination).stars, result.stars)
    pd.testing.assert_frame_equal(stars, original)
    with pytest.raises(FileExistsError, match="load_target"):
        prepare_target(saved, destination)
    (destination/"stars.csv").write_text("altered")
    with pytest.raises(ValueError, match="changed"):
        load_target(destination)


def test_query_download_and_population_reader_round_trip(monkeypatch, tmp_path, inputs):
    saved, stars, _ = inputs
    calls = []
    def catalog(host_id, radius):
        calls.append((host_id, radius))
        return stars
    monkeypatch.setattr(field, "_query_stars", catalog)
    host = stars[stars.ID == "123"].iloc[0]
    def query(ra, dec, **kwargs):
        np.testing.assert_allclose([ra, dec], [host.ra, host.dec])
        assert kwargs["mag_lim"] == "21.0"
        return "https://fixture.invalid/population"
    monkeypatch.setattr("pentaceratops.stellar.query_TRILEGAL", query)
    responses = [b"still running", RAW_POPULATION]
    monkeypatch.setattr(field, "urlopen", lambda *a, **k: io.BytesIO(responses.pop(0)))
    monkeypatch.setattr(field.time, "sleep", lambda s: None)
    cwd = Path.cwd()
    target = prepare_target(saved, tmp_path/"field")
    assert Path.cwd() == cwd
    assert calls == [(123, 10)] and not responses
    # Exercise the actual evidence population parser; no real stars lost to footer slicing.
    magnitudes, masses, _, _, _, _, _, _ = trilegal_results(target.trilegal_fname, 10.)
    np.testing.assert_array_equal(magnitudes, [14., 16.])
    np.testing.assert_array_equal(masses, [1., .8])
    assert target.field_metadata["population"]["source"] == "TRILEGAL"


def test_adopted_single_footer_population_is_copied_without_changing_reader_selection(monkeypatch, tmp_path, inputs):
    saved, stars, population = inputs
    no_network(monkeypatch)
    table = pd.read_csv(population).drop(index=2)  # Real service files may have only the final footer.
    table.to_csv(population, index=False)
    expected = trilegal_results(population, 10.)
    target = prepare_target(saved, tmp_path/"field", stars=stars, trilegal_fname=population)
    assert Path(target.trilegal_fname).read_bytes() == population.read_bytes()
    for actual, reference in zip(trilegal_results(target.trilegal_fname, 10.), expected):
        np.testing.assert_array_equal(actual, reference)
    # This test freezes the adopted reader behavior, not a reinterpretation of its input.
    assert len(expected[0]) == 1


@pytest.mark.parametrize("damage", ["mission", "depth", "fits", "id", "missing_mass", "empty_population"])
def test_bad_inputs_fail_without_population_download(monkeypatch, tmp_path, inputs, damage):
    saved, stars, population = inputs
    no_network(monkeypatch)
    if damage == "mission":
        saved.metadata["settings"]["candidate"]["mission"] = "Kepler"
    elif damage == "depth":
        del saved.output["windows"]["transit_depth"]
    elif damage == "fits":
        saved.metadata["settings"]["native_products"][0]["fits_sha256"] = "bad"
    elif damage == "id":
        stars.loc[stars.ID == "123", "ID"] = "123.5"
    elif damage == "missing_mass":
        stars.loc[stars.ID == "123", "mass"] = np.nan
    elif damage == "empty_population":
        population.write_text("placeholder\n")
    with pytest.raises(ValueError):
        prepare_target(saved, tmp_path/"field", stars=stars, trilegal_fname=population)
    assert not (tmp_path/"field").exists()


def test_explicit_fractional_depth_and_csv_catalog(monkeypatch, tmp_path, inputs):
    saved, stars, population = inputs
    no_network(monkeypatch)
    del saved.output["windows"]["transit_depth"]
    catalog = tmp_path/"catalog.csv"
    stars.to_csv(catalog, index=False)
    target = prepare_target(saved, tmp_path/"field", stars=catalog,
                            trilegal_fname=population, transit_depth=.02)
    np.testing.assert_allclose(target.stars.tdepth*target.stars.fluxratio, .02)
    assert target.field_metadata["depth_source"] == "explicit"


def test_missing_neighbor_photometry_survives_preparation_and_reload(monkeypatch, tmp_path, inputs):
    saved, stars, population = inputs
    no_network(monkeypatch)
    columns = ["plx", "Jmag", "Hmag", "Kmag"]
    stars.loc[stars.ID == "456", columns] = np.nan
    target = prepare_target(saved, tmp_path/"field", stars=stars, trilegal_fname=population)
    for host in (target, load_target(tmp_path/"field")):
        assert host.stars.ID.tolist() == [123, 456]
        assert host.stars.iloc[1].tdepth > 0
        assert host.stars.loc[1, columns].isna().all()


def test_missing_neighbor_mass_remains_an_error(monkeypatch, tmp_path, inputs):
    saved, stars, population = inputs
    no_network(monkeypatch)
    stars.loc[stars.ID == "456", "mass"] = np.nan
    with pytest.raises(ValueError, match="Invalid mass for host 456"):
        prepare_target(saved, tmp_path/"field", stars=stars, trilegal_fname=population)
    assert not (tmp_path/"field").exists()


def test_broken_output_symlink_is_not_followed(monkeypatch, tmp_path, inputs):
    saved, _, _ = inputs
    no_network(monkeypatch)
    destination = tmp_path/"field"
    destination.symlink_to(tmp_path/"elsewhere")
    with pytest.raises(FileExistsError):
        prepare_target(saved, destination)
    assert not (tmp_path/"elsewhere").exists()


@pytest.mark.parametrize("failure", ["unavailable", "timeout", "missing_TESS"])
def test_population_service_failure_never_publishes_partial_field(monkeypatch, tmp_path, inputs, failure):
    saved, stars, _ = inputs
    monkeypatch.setattr("pentaceratops.stellar.query_TRILEGAL",
        lambda *a, **k: None if failure == "unavailable" else "https://fixture.invalid/population")
    payload = RAW_POPULATION.replace(b"TESS", b"Vmag") if failure == "missing_TESS" else b"running"
    monkeypatch.setattr(field, "urlopen", lambda *a, **k: io.BytesIO(payload))
    ticks = iter([0., 0., 2., 2.])
    monkeypatch.setattr(field.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(field.time, "sleep", lambda _: None)
    with pytest.raises((RuntimeError, TimeoutError, ValueError)):
        prepare_target(saved, tmp_path/"field", stars=stars, population_timeout=1.)
    assert not (tmp_path/"field").exists()
    assert not list(tmp_path.glob(".pentaceratops-field-*"))


@pytest.mark.parametrize("domain", ["real", "fourier"])
def test_prepared_field_runs_evidence(monkeypatch, tmp_path, inputs, domain):
    saved, stars, population = inputs
    no_network(monkeypatch)
    target = prepare_target(saved, tmp_path/"field", stars=stars, trilegal_fname=population)
    result = evidence(saved, target=target, likelihood=domain, N=8, steps=1,
        nsamples=3, posterior_samples=9, scenarios=["TP", "EB", "NTP"], seed=307)
    assert len(result.output) == 3
    assert np.isfinite(result.output.lnBF).all()
