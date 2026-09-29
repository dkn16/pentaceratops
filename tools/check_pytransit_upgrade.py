#!/usr/bin/env python3
"""Save/compare small cross-environment fixtures; no catalogues or production fits.

Run ``snapshot FILE.npz`` once with each dependency stack, then
``compare BEFORE.npz AFTER.npz``. Artifacts belong on scratch, not in Git.
Same-environment reference tests alone cannot detect dependency-induced drift.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from pentaceratops import RunResult, run_evidence
from pentaceratops.likelihoods import real


def snapshot(path):
    time = np.linspace(-.25, .25, 101)
    other = np.linspace(-.24, .24, 97)
    outputs, parameters = {}, {}
    # Include short/long periods, eccentric, grazing, near-equal radii, and
    # different dilution/host assignments. These are numerical fixtures.
    for case, (period, ecc, argp, impact, radius) in enumerate([
        (.75, 0., 90., .1, .1), (2., .6, 90., .2, .5),
        (5., .02, 65., .85, 1.), (10., .1, 240., .3, .99999999),
        (27., .1, 310., .9, 1.1), (159., .4, 180., .4, .2),
    ]):
        a = (real.G*real.Msun*(period*86400)**2/(4*np.pi**2))**(1/3)
        cosi = impact*real.Rsun/a*(1+ecc*np.sin(np.deg2rad(argp)))/(1-ecc**2)
        common = dict(P_orb=period, inc=np.rad2deg(np.arccos(cosi)), a=a,
                      R_s=1., u1=.3, u2=.2, ecc=ecc, argp=argp,
                      companion_fluxratio=.2, companion_is_host=bool(case % 2))
        for minutes in (2, 30):
            for count in (1, 5, 20, 50):
                key = f"case{case}_exp{minutes}_n{count}"
                settings = dict(common, exptime=minutes/1440., nsamples=count)
                parameters[key] = dict(settings, R_EB=radius, R_p=1+case*3.)
                outputs[key+"_planet"] = real.simulate_TP_transit(time, R_p=1+case*3., **settings)
                eb = dict(settings, R_EB=radius, EB_fluxratio=.25)
                primary, secondary = real.simulate_EB_transit_secondary(time, other, **eb)
                outputs[key+"_eb_primary"] = primary
                outputs[key+"_eb_secondary"] = secondary
                even, odd = real.simulate_EB_transit_evenodd(time, other, **dict(eb, P_orb=2*period))
                outputs[key+"_x2p_even"] = even
                outputs[key+"_x2p_odd"] = odd
    # Reuse the tiny seeded evidence fixture used by the migration suite.
    # This is a smoke check, not a claim of high-precision evidence stability.
    for domain in ("real", "fourier"):
        module = __import__("pentaceratops.evidence."+domain, fromlist=["*"])
        name = "lnZ_TTP" if domain == "real" else "lnZ_TTP_fourier"
        t = np.linspace(-.12, .12, 31)
        kwargs = dict(time=t, flux=1-.0005*np.exp(-.5*(t/.035)**2), sigma=.003,
                      P_orb=3., M_s=.8, R_s=.75, Teff=4800., Z=0., N=8, steps=2,
                      mission="TESS", exptime=.002, nsamples=3)
        if domain == "fourier":
            kwargs["var_fourier"] = np.full(len(t)//2, len(t)*.003**2)
        result = run_evidence(getattr(module, name), seed=73, **kwargs)
        outputs[domain+"_evidence"] = result.output
    versions = {name: importlib.metadata.version(name)
                for name in ("pytransit", "meepmeep", "numpy", "numba", "setuptools")}
    RunResult(outputs, metadata=dict(artifact="pytransit_upgrade_fixture", schema=1,
        python=platform.python_version(), versions=versions, parameters=parameters,
        time=time.tolist(), secondary_time=other.tolist())).save(path)
    print(json.dumps(dict(path=str(path), versions=versions), indent=2))


def compare(before, after):
    a, b = RunResult.load(before), RunResult.load(after)
    for key in ("artifact", "schema", "parameters", "time", "secondary_time"):
        if a.metadata[key] != b.metadata[key]:
            raise ValueError(f"Fixture settings differ: {key}")
    if a.output.keys() != b.output.keys():
        raise ValueError("Fixture output keys differ")
    model_deltas, evidence = {}, {}
    for key, left in a.output.items():
        right = b.output[key]
        if key.endswith("_evidence"):
            evidence[key] = dict(before_lnZ=np.asarray(left["lnZ"]).tolist(),
                                after_lnZ=np.asarray(right["lnZ"]).tolist(),
                                delta_lnZ=(np.asarray(right["lnZ"])-np.asarray(left["lnZ"])).tolist())
        else:
            np.testing.assert_array_equal(np.isfinite(left), np.isfinite(right))
            finite = np.isfinite(left)
            if not finite.any():
                raise ValueError(f"All nonfinite values: {key}")
            model_deltas[key] = float(np.max(abs(left[finite]-right[finite]))*1e6)
    groups = {}
    for group in ("planet", "eb_primary", "eb_secondary", "x2p_even", "x2p_odd"):
        selected = {k:v for k,v in model_deltas.items() if k.endswith("_"+group)}
        worst = max(selected, key=selected.get)
        groups[group] = dict(max_abs_ppm=selected[worst], worst_fixture=worst,
                             median_max_abs_ppm=float(np.median(list(selected.values()))))
    print(json.dumps(dict(before=a.metadata["versions"], after=b.metadata["versions"],
                          model_comparisons=len(model_deltas), groups=groups,
                          evidence=evidence), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    save = sub.add_parser("snapshot")
    save.add_argument("path", type=Path)
    diff = sub.add_parser("compare")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.path)
    else:
        compare(args.before, args.after)


if __name__ == "__main__":
    main()
