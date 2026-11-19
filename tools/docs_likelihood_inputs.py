"""Plot what each likelihood is given, for the likelihoods page of the documentation.

  subtracted likelihood  : the folded even, odd and secondary windows after the variability is removed
  marginalized likelihood: every observed bin folded at twice the period, and the correlation of its noise

Usage: python tools/docs_likelihood_inputs.py FOLDED_REAL.npz FOLDED_FOURIER_DIR OUTPUT_DIR
Writes likelihood_input_{subtracted,marginalized}_{light,dark}.svg into OUTPUT_DIR.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781", grid="#e1e0d9", data="#2a78d6"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#898781", grid="#2c2c2a", data="#3987e5"),
}


def style(theme):
    plt.rcParams.update({
        "font.size": 10.5, "font.family": "DejaVu Sans", "axes.edgecolor": theme["muted"], "axes.linewidth": .8,
        "axes.labelcolor": theme["text2"], "xtick.color": theme["muted"], "ytick.color": theme["muted"],
        "xtick.labelcolor": theme["text2"], "ytick.labelcolor": theme["text2"], "axes.titlecolor": theme["text"],
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "svg.fonttype": "path"})


def tidy(ax, theme):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=theme["grid"], lw=.6); ax.set_axisbelow(True)


def subtracted(data, theme, path):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharey=True)
    fig.subplots_adjust(left=.125, right=.985, bottom=.17, top=.88, wspace=.08)
    for ax, panel, title in zip(axes, ("even", "odd", "secondary"), ("Even transits", "Odd transits", "Secondary phase")):
        t, f, s = 24*data[panel+"_time"], 1e6*(data[panel+"_mean_flux"]-1), 1e6*data[panel+"_sigma"]
        ax.errorbar(t, f, yerr=s, fmt="o", ms=3.6, color=theme["data"], ecolor=theme["muted"], elinewidth=.8, lw=0)
        ax.set_title(title, fontsize=11.5, weight="bold", loc="left")
        ax.set_xlabel("Hours from window centre"); tidy(ax, theme)
    axes[0].set_ylabel("Relative flux [ppm]")
    fig.savefig(path); plt.close(fig)
    return len(data["even_time"]), 1440*float(data["exptime"])


def marginalized(folded, theme, path):
    block, period = folded.block, float(folded.period)
    shown = (block.time-folded.epoch+period/2) % (2*period)-period/2      # transits at 0 and at one period
    order = np.argsort(shown)
    fig = plt.figure(figsize=(8.4, 5.4))
    top = fig.add_axes([.125, .52, .86, .40]); bottom = fig.add_axes([.125, .105, .86, .21])
    top.errorbar(shown[order], 1e6*(block.flux[order]-1), yerr=1e6*block.sigma[order], fmt="o", ms=3.0,
                 color=theme["data"], ecolor=theme["muted"], elinewidth=.6, lw=0)
    top.set_xlim(-period/2, 3*period/2)
    top.set_title("Every observed bin, folded at twice the candidate period", fontsize=11.5, weight="bold", loc="left")
    top.set_xlabel("Time since the reference transit [days]"); top.set_ylabel("Relative flux [ppm]"); tidy(top, theme)
    low, high = top.get_ylim(); span = high-low
    top.set_ylim(low, high+.22*span)                       # headroom for the labels
    for x, label in ((0, "even transit"), (period, "odd transit"), (period/2, "secondary phase")):
        top.annotate(label, (x, high+.11*span), ha="center", va="center", color=theme["text2"], fontsize=9.5)
    # correlation of the noise between one out-of-transit bin and the others, against their separation in the fold
    covariance = np.linalg.inv(block.precision)
    centre = int(np.argmin(np.abs(shown-period/2)))
    lag = 24*(shown-shown[centre])
    correlation = covariance[centre]/np.sqrt(covariance[centre, centre]*np.diag(covariance))
    keep = np.argsort(lag); keep = keep[np.abs(lag[keep]) <= 24]
    bottom.plot(lag[keep], correlation[keep], "-", color=theme["data"], lw=2)
    bottom.set_xlim(-24, 24); bottom.axhline(0, color=theme["muted"], lw=.8)
    bottom.set_title("Its noise is correlated: correlation between two bins", fontsize=11.5, weight="bold", loc="left")
    bottom.set_xlabel("Separation of the two bins [hours]"); bottom.set_ylabel("Correlation"); tidy(bottom, theme)
    fig.savefig(path); plt.close(fig)
    near = np.abs(lag) > 0
    return len(block.time), float(np.median(1e6*block.sigma)), float(correlation[near][np.argmin(np.abs(lag[near]))])


def main():
    from pentaceratops import RunResult, FoldedFourierData
    data = RunResult.load(sys.argv[1]).output
    folded = FoldedFourierData.load(sys.argv[2])
    out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        style(theme)
        a = subtracted(data, theme, out/f"likelihood_input_subtracted_{name}.svg")
        b = marginalized(folded, theme, out/f"likelihood_input_marginalized_{name}.svg")
    print("subtracted: bins per window, minutes per bin:", a)
    print("marginalized: bins, median sigma [ppm], correlation of adjacent bins:", b)


if __name__ == "__main__":
    main()
