"""Animate one persistent-sampling run recorded by tools/docs_sampling_run.py.

Left: every particle kept so far in the (planet radius, impact parameter) plane; marker area is its
current importance weight. Right: the folded transit data with light curves drawn from the weighted pool.
Each stage raises the inverse temperature beta, reweights the whole pool, then adds one new population.

Usage: python tools/docs_sampling_animation.py RUN.npz OUTPUT_STEM [light|dark]
Writes OUTPUT_STEM.webm (VP9) and OUTPUT_STEM.png (last frame).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781", grid="#e1e0d9",
                  kept="#2a78d6", new="#eb6834"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 kept="#3987e5", new="#d95926"),
}
FPS, REWEIGHT, ADD, HOLD_START, HOLD_END = 15, 12, 9, 14, 45
N_CURVES = 45
INK_AREA = 14.0          # marker area [pt^2] of one particle when all weights are equal in the first population


def main():
    run, stem = sys.argv[1], sys.argv[2]
    theme = THEMES[sys.argv[3] if len(sys.argv) > 3 else "light"]
    z = np.load(run)
    N = int(z["N"]); betas = z["betas"]; lnZ = z["lnZ"]; sizes = z["pool_sizes"]
    rp, b, models, grid = z["rp"], z["b"], z["models"], z["grid"]
    stages = len(betas)
    # normalized pool weights at every stage, padded with zeros for particles not created yet
    weights = np.zeros((stages, len(rp)))
    start = 0
    for k, n in enumerate(sizes):
        lw = z["log_w"][start:start+n]; start += n
        w = np.exp(lw-lw.max()); weights[k, :n] = w/w.sum()
    rng = np.random.default_rng(3)
    curves = [rng.choice(len(rp), N_CURVES, p=weights[k]) for k in range(stages)]

    plt.rcParams.update({"font.size": 12.5, "font.family": "DejaVu Sans", "axes.edgecolor": theme["muted"],
                         "axes.labelcolor": theme["text2"], "xtick.color": theme["muted"], "ytick.color": theme["muted"],
                         "xtick.labelcolor": theme["text2"], "ytick.labelcolor": theme["text2"],
                         "figure.facecolor": theme["surface"], "axes.facecolor": theme["surface"],
                         "savefig.facecolor": theme["surface"], "axes.linewidth": .8})
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    left = fig.add_axes([.075, .135, .40, .60]); right = fig.add_axes([.565, .135, .40, .60])
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=theme["grid"], lw=.6); ax.set_axisbelow(True)

    def heading(x, bold, rest):
        """Bold panel title followed by a lighter subtitle on the same line."""
        first = fig.text(x, .775, bold, color=theme["text"], fontsize=14.5, weight="bold")
        width = first.get_window_extent(fig.canvas.get_renderer()).width/fig.bbox.width
        fig.text(x+width+.012, .775, rest, color=theme["text2"], fontsize=12)

    # ---- left: particles
    left.set_xscale("log"); left.set_xlim(.45, 22); left.set_ylim(-.03, 1.40); left.set_yticks(np.arange(0, 1.01, .2))
    left.set_xticks([.5, 1, 2, 5, 10, 20]); left.set_xticklabels(["0.5", "1", "2", "5", "10", "20"]); left.minorticks_off()
    left.set_xlabel("Planet radius [Earth radii]"); left.set_ylabel("Impact parameter")
    faint = left.scatter([], [], s=2.2, color=theme["muted"], alpha=.30, lw=0, zorder=2)
    kept = left.scatter([], [], s=[], color=theme["kept"], alpha=.62, lw=0, zorder=3)
    new = left.scatter([], [], s=[], color=theme["new"], alpha=0., lw=0, zorder=4)
    heading(.075, "Particles", "marker area = current weight")
    left.scatter([], [], s=45, color=theme["kept"], label="kept from earlier stages")
    left.scatter([], [], s=45, color=theme["new"], label="added at this stage")
    left.legend(loc="upper right", frameon=False, fontsize=11.5, labelcolor=theme["text2"], handletextpad=.3,
                borderaxespad=.2)
    median_text = left.text(.03, .965, "", transform=left.transAxes, color=theme["text"], fontsize=12, va="top",
                            bbox=dict(facecolor=theme["surface"], edgecolor="none", alpha=.85, pad=3), zorder=6)

    # ---- right: light curve
    hours = 24*grid
    t = 24*np.concatenate([z["t_even"], z["t_odd"]]); f = 1e6*(np.concatenate([z["f_even"], z["f_odd"]])-1)
    s = 1e6*np.concatenate([z["s_even"], z["s_odd"]])
    right.set_xlim(-6.2, 6.2); right.set_ylim(-1700, 900)
    right.set_xlabel("Time from mid-transit [hours]"); right.set_ylabel("Relative flux [ppm]")
    right.errorbar(t, f, yerr=s, fmt="o", ms=3.2, color=theme["text2"], ecolor=theme["muted"], elinewidth=.7,
                   alpha=.85, zorder=5, lw=0)
    lines_a = [right.plot(hours, hours*0, color=theme["kept"], lw=1.0, alpha=0., zorder=3)[0] for _ in range(N_CURVES)]
    lines_b = [right.plot(hours, hours*0, color=theme["kept"], lw=1.0, alpha=0., zorder=3)[0] for _ in range(N_CURVES)]
    heading(.565, "What they predict", "transit data and models from the pool")

    # ---- header: stage, beta, evidence, caption, progress
    title = fig.text(.075, .925, "", color=theme["text"], fontsize=17, weight="bold")
    caption = fig.text(.075, .868, "", color=theme["text2"], fontsize=13.5)
    numbers = fig.text(.965, .928, "", color=theme["text2"], fontsize=12.5, ha="right", family="DejaVu Sans Mono")
    bar_ax = fig.add_axes([.075, .845, .89, .008]); bar_ax.set_axis_off(); bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1)
    bar_ax.barh(.5, 1, height=1, color=theme["grid"])
    progress = bar_ax.barh(.5, 0, height=1, color=theme["kept"])[0]

    def fmt_beta(x):
        return "0" if x < 5e-7 else ("1" if x >= 1 else (f"{x:.4f}" if x < .01 else f"{x:.3f}"))

    def draw(k, w, n_pool, new_alpha, curve_mix, caption_text, frac):
        """k: stage; w: weights to show for the first n_pool particles; new population is [n_pool, n_pool+N)."""
        idx = np.arange(n_pool)
        faint.set_offsets(np.column_stack([rp[idx], b[idx]]))
        area = np.clip(INK_AREA*N*w[idx], 0, 160)
        show = area > .15
        kept.set_offsets(np.column_stack([rp[idx][show], b[idx][show]])); kept.set_sizes(area[show])
        if new_alpha > 0 and n_pool+N <= len(rp):
            j = np.arange(n_pool, n_pool+N)
            new.set_offsets(np.column_stack([rp[j], b[j]])); new.set_sizes(np.full(N, 11.)); new.set_alpha(.75*new_alpha)
        else:
            new.set_alpha(0.)
        order = np.argsort(rp[idx]); cw = np.cumsum(w[idx][order]); med = rp[idx][order][np.searchsorted(cw, .5*cw[-1])]
        median_text.set_text(f"weighted median radius\n{med:.2f} Earth radii")
        ka, kb, mix = curve_mix
        for line, i in zip(lines_a, curves[ka]):
            line.set_ydata(1e6*(models[i]-1)); line.set_alpha(.30*(1-mix))
        for line, i in zip(lines_b, curves[kb]):
            line.set_ydata(1e6*(models[i]-1)); line.set_alpha(.30*mix)
        title.set_text(f"Persistent sampling: stage {k+1} of {stages}")
        caption.set_text(caption_text)
        numbers.set_text(f"β = {fmt_beta(betas[k]):<6s}  {n_pool:>4d} particles kept   ln Z(β) = {lnZ[k]:6.2f}")
        progress.set_width(frac)

    frames = []
    for k in range(stages):
        n_pool = int(sizes[k]); prev = max(k-1, 0)
        if k == 0:
            frames += [(k, weights[0], n_pool, 0., (0, 0, 1.), f"Start with {N} draws from the prior (β = 0).", 0.)]*HOLD_START
        else:
            for i in range(1, REWEIGHT+1):
                m = i/REWEIGHT; m = m*m*(3-2*m)
                frames.append((k, (1-m)*weights[prev]+m*weights[k], n_pool, 0., (prev, k, m),
                               f"Raise β from {fmt_beta(betas[prev])} to {fmt_beta(betas[k])}: every particle kept so far is reweighted.",
                               (k-1+.5*m)/(stages-1)))
        if k < stages-1:
            for i in range(1, ADD+1):
                frames.append((k, weights[k], n_pool, i/ADD, (k, k, 1.),
                               f"Resample from the weighted pool and move {N} new particles with MCMC. Nothing is thrown away.",
                               (k+.5+.5*i/ADD)/(stages-1) if k else .5*i/ADD/(stages-1)))
        else:
            frames += [(k, weights[k], n_pool, 0., (k, k, 1.),
                        "β = 1: the weighted pool is the posterior, and its normalization is the evidence.", 1.)]*HOLD_END

    import shutil, subprocess, tempfile
    folder = Path(tempfile.mkdtemp(prefix="frames_", dir=Path(stem).resolve().parent))
    try:
        for i, frame in enumerate(frames):
            draw(*frame); fig.savefig(folder/f"{i:05d}.png", dpi=100)
        fig.savefig(stem+".png", dpi=100)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(folder/"%05d.png"),
                        "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "34", "-pix_fmt", "yuv420p",
                        "-an", stem+".webm"], check=True)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    print(len(frames), "frames,", round(len(frames)/FPS, 1), "s")


if __name__ == "__main__":
    main()
