"""Run one target-planet fit on real folded data and keep what the sampler did at every stage.
Usage: python tools/docs_sampling_run.py FOLDED_REAL.npz FIELD_DIR OUTPUT.npz [N] [STEPS]"""
import sys
import numpy as np


def main():
    folded, field, output = sys.argv[1:4]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    from scipy.special import logsumexp
    from pentaceratops import RunResult, load_target
    from pentaceratops.hz import real_adapter
    from pentaceratops.sampling import persistent
    from pentaceratops.evidence.joint_fourier import sample_joint_scenario
    from pentaceratops.models.systems import semimajor_axis, RSUN

    data = RunResult.load(folded).output
    target = load_target(field)
    star = target.stars.iloc[0].copy()
    adapter = real_adapter(data, mission=target.mission, nsamples=20, timing_policy="observed")
    adapter.aperture_fraction = float(star.fluxratio)

    stages = []                      # one entry per temperature step: beta, pool weights, evidence
    original = persistent.solve_for_next_beta

    def record(current_beta, log_ls, log_S, target_ess, history_len):
        beta, log_w = original(current_beta, log_ls, log_S, target_ess, history_len)
        stages.append(dict(beta=float(beta), log_w=np.array(log_w), lnZ=float(logsumexp(log_w)-np.log(len(log_w)))))
        return beta, log_w
    persistent.solve_for_next_beta = record
    try:
        result = sample_joint_scenario(adapter, "TP", star, trilegal=target.trilegal_fname, N=N, steps=steps,
                                       seed=42, posterior_samples=2000)
    finally:
        persistent.solve_for_next_beta = original

    pool = result["physical_parameters"]          # period, inc, ecc, argp, R_p ; N rows per stage
    period, inc, ecc, argp, rp = pool.T
    a = semimajor_axis(period, float(star.mass))
    b = a*(1-ecc**2)/(1+ecc*np.sin(np.radians(argp)))*np.cos(np.radians(inc))/(float(star.rad)*RSUN)
    best = result["bestfit"]["parameters"]
    # light curves of every pool particle on a fine grid, in the aperture frame
    lk = adapter.lk
    half = float(max(abs(adapter.times["even"]).max(), abs(adapter.times["odd"]).max()))
    grid = np.linspace(-half, half, 241)
    models = np.empty((len(pool), len(grid)), "f4")
    for i in range(len(pool)):
        p = dict(best, R_p=rp[i], inc=inc[i], ecc=ecc[i], argp=argp[i], a=a[i])
        p.pop("epoch_offset", None)
        models[i] = 1+adapter.aperture_fraction*(lk.simulate_TP_transit(grid, **p)-1)
    lk._tm_cache["id_time"] = None
    np.savez(output, N=N, steps=steps, rp=rp, b=b, ecc=ecc, inc=inc, weights=result["weights"], log_target=result["log_target"],
             betas=np.array([s["beta"] for s in stages]), lnZ=np.array([s["lnZ"] for s in stages]),
             pool_sizes=np.array([len(s["log_w"]) for s in stages]),
             log_w=np.concatenate([s["log_w"] for s in stages]), lnBF=result["lnBF"],
             grid=grid, models=models,
             t_even=adapter.times["even"], f_even=data["even_mean_flux"], s_even=data["even_sigma"],
             t_odd=adapter.times["odd"], f_odd=data["odd_mean_flux"], s_odd=data["odd_sigma"],
             best_rp=best["R_p"], period=float(period[0]))
    print("pool", pool.shape, "stages", len(stages), "betas", np.round([s["beta"] for s in stages], 5), "lnBF", result["lnBF"])
    print("pool sizes", [len(s["log_w"]) for s in stages])
    print("posterior R_p", np.average(rp, weights=result["weights"]), "b", np.average(b, weights=result["weights"]))


if __name__ == "__main__":
    main()
