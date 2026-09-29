import numpy as np
import pandas as pd
import pytest

from pentaceratops.results import RunResult, scenario_probabilities


def test_standard_fpp_is_distinct_from_eb_only():
    table = pd.DataFrame({"scenario": ["TP", "STP", "BTP", "EB", "NTP"],
                          "lnBF": np.log([4, 1, 1, 2, 2])})
    out = scenario_probabilities(table, evidence_column="lnBF")
    assert out.attrs["FPP"] == pytest.approx(.6)
    assert out.attrs["FPP_EB"] == pytest.approx(.2)
    assert out.attrs["NFPP"] == pytest.approx(.2)
    assert "prob" not in table


def test_eta_is_applied_once_to_log_evidence_not_existing_probabilities():
    table = pd.DataFrame({"scenario": ["TP", "EB", "BEBx2P"], "lnZ": [1e6]*3})
    out = scenario_probabilities(table, evidence_column="lnZ", eb_eta=.1)
    second = scenario_probabilities(out, evidence_column="lnZ", eb_eta=.1)
    np.testing.assert_allclose(out.prob, [1/1.2, .1/1.2, .1/1.2])
    np.testing.assert_array_equal(out.prob, second.prob)


def test_large_common_offset_does_not_erase_prior_odds():
    out = scenario_probabilities({"scenario": ["TP", "EB"], "lnZ": [1e300, 1e300]},
                                 evidence_column="lnZ", eb_eta=.1)
    np.testing.assert_allclose(out.prob, [1/1.1, .1/1.1], rtol=1e-14)


@pytest.mark.parametrize("values", [[-np.inf, -np.inf], [np.nan, 0], [np.inf, 0]])
def test_bad_evidence_is_not_silently_excluded(values):
    with pytest.raises(ValueError):
        scenario_probabilities({"scenario": ["TP", "EB"], "lnZ": values}, evidence_column="lnZ")


@pytest.mark.parametrize("eta", [-1, np.inf, np.nan])
def test_bad_prior(eta):
    with pytest.raises(ValueError):
        scenario_probabilities({"scenario": ["TP"], "lnZ": [0]}, evidence_column="lnZ", eb_eta=eta)


def test_zero_prior_and_impossible_evidence():
    out = scenario_probabilities({"scenario": ["TP", "EB", "NEB"], "lnZ": [0, 1, -np.inf]},
                                 evidence_column="lnZ", eb_eta=0)
    np.testing.assert_array_equal(out.prob, [1, 0, 0])
    assert out.attrs["FPP"] == 0


def test_unknown_scenario():
    with pytest.raises(ValueError, match="Unknown"):
        scenario_probabilities({"scenario": ["typo"], "lnZ": [0]}, evidence_column="lnZ")


def test_explicit_evidence_column_required():
    with pytest.raises(TypeError):
        scenario_probabilities({"scenario": ["TP"], "lnZ": [0]})


def test_bundle_roundtrip_no_pickle_no_overwrite(tmp_path):
    table = pd.DataFrame({"scenario": ["TP", "EB"], "lnBF": [2., -np.inf], "R_p": [1., np.nan]})
    table.attrs["FPP"] = .01
    result = RunResult(table, [{"kind": "sampler", "samples": np.arange(12).reshape(4, 3),
                               "weights": np.ones(4)/4, "log_target": np.arange(4.)}],
                       {"note": "portable", "seed": np.int64(17)})
    path = result.save(tmp_path / "posterior.npz")
    with np.load(path, allow_pickle=False) as raw:
        assert all(not raw[key].dtype.hasobject for key in raw.files)
    loaded = RunResult.load(path)
    pd.testing.assert_frame_equal(loaded.output, result.output)
    assert loaded.output.attrs == table.attrs
    np.testing.assert_array_equal(loaded.sampling[0]["samples"], result.sampling[0]["samples"])
    assert loaded.metadata == result.metadata
    with pytest.raises(FileExistsError):
        result.save(path)
    assert list(tmp_path.iterdir()) == [path]


def test_object_array_rejected(tmp_path):
    with pytest.raises(TypeError):
        RunResult(np.array([object()], dtype=object)).save(tmp_path / "bad.npz")
    assert not list(tmp_path.iterdir())


def test_tuple_roundtrip(tmp_path):
    original = (np.arange(3), {"lnZ": -1.})
    loaded = RunResult.load(RunResult(original).save(tmp_path / "tuple.npz"))
    assert isinstance(loaded.output, tuple)
    np.testing.assert_array_equal(loaded.output[0], original[0])
    assert loaded.output[1] == original[1]


def test_named_host_index_survives_storage(tmp_path):
    table = pd.DataFrame({"prob": [.3, .7]}, index=pd.Index([104, 109], name="host_id"))
    loaded = RunResult.load(RunResult(table).save(tmp_path / "hosts.npz"))
    pd.testing.assert_frame_equal(table, loaded.output)
