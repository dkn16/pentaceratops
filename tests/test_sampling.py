import numpy as np
import pytest

from pentaceratops import run_evidence
from pentaceratops.sampling.persistent import persistent_sampling
from pentaceratops.sampling.policy import SCENARIOS, sampling_config, sampling_policy
from pentaceratops.sampling.records import capture_sampling, record_evidence


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_policy_override_and_copy(scenario):
    default = sampling_config(scenario)
    assert sampling_config(scenario, N=9)["steps"] == default["steps"]
    assert sampling_config(scenario, steps=4)["N"] == default["N"]
    altered = sampling_policy()
    altered[scenario]["N"] = -100
    assert sampling_config(scenario)["N"] > 0


@pytest.mark.parametrize("value", [0, -1, True, 1.2])
def test_invalid_policy_override(value):
    with pytest.raises(ValueError):
        sampling_config("TP", N=value)


def test_recording_preserves_rng_and_full_pool(tmp_path):
    kwargs = dict(log_likelihood_function=lambda x: -2*x[0]**2,
                  prior_transform=lambda x: 2*x-1, n_active=20, ndim=1,
                  target_ess=40, mcmc_steps=2)
    np.random.seed(731)
    expected = persistent_sampling.__wrapped__(**kwargs)
    next_expected = np.random.random()
    np.random.seed(731)
    with capture_sampling() as records:
        actual = persistent_sampling(**kwargs)
    next_actual = np.random.random()
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)
    assert next_expected == next_actual
    assert len(records) == 1
    np.testing.assert_array_equal(records[0]["samples"], actual[2])
    np.testing.assert_array_equal(records[0]["log_target"], actual[3])
    assert records[0]["weights"].sum() == pytest.approx(1)
    assert len(records[0]["weights"]) >= 20
    result = run_evidence(persistent_sampling, seed=731,
                          output_path=tmp_path / "run.npz", **kwargs)
    np.testing.assert_array_equal(result.sampling[0]["weights"], actual[1])


def test_explicit_seed_restores_caller_random_state():
    np.random.seed(18)
    expected = np.random.random()
    np.random.seed(18)
    one = run_evidence(np.random.random, seed=22).output
    assert np.random.random() == expected
    assert run_evidence(np.random.random, seed=22).output == one


def test_failed_evidence_record_and_context_cleanup():
    @record_evidence
    def fails():
        raise ValueError("bad input")

    with capture_sampling() as records:
        with pytest.raises(ValueError):
            fails()
    assert records[0]["status"] == "failed"
    with capture_sampling() as next_records:
        assert next_records == []
