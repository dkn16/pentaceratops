"""Full-orbit folding with the same observation operator for data and models."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from ..experimental.folded_joint import full_orbit_fold, FoldedJointFourierAdapter
from ..likelihoods.joint_fourier import GaussianBlock, observed_covariance
from .joint_fourier import save_block, load_block


def load_folded_real(prepared_path, posterior_path, *, primary_only=False):
    """Load the paired HZ preparation and conditional-FGP posterior caches.

    Preserve the adopted per-panel median measurement errors and the common
    coefficient ordering of the cross-panel covariance factor. Primary-only
    mode requires a cache explicitly prepared without secondary observations.
    """
    from ..experimental.covariance import load_inputs
    from ..experimental.primary_v2 import load_primary_inputs
    row = dict(prepared_path=prepared_path, folded_path=posterior_path)
    loader = load_primary_inputs if primary_only else load_inputs
    return loader(row, "")


@dataclass
class FoldedFourierData:
    """An observed 2P fold and its native exposure-to-bin map.

    The native times are retained because a bin-center model does not generally
    equal the mean of the contributing exposure-integrated models.
    """
    block: GaussianBlock
    native_time: np.ndarray
    bin_index: np.ndarray
    weights: np.ndarray
    period: float
    epoch: float

    def adapter(self, *, mission, filt=None, parity="profile", nsamples=7):
        return FoldedJointFourierAdapter(
            self.block, self.native_time, self.bin_index, self.weights,
            self.period, self.epoch, mission=mission, filt=filt,
            parity=parity, nsamples=nsamples,
        )

    def save(self, directory):
        """Save precision separately so scenario workers can memory-map it."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        save_block(self.block, directory / "block")
        np.savez_compressed(directory / "fold_map.npz", native_time=self.native_time,
                            bin_index=self.bin_index, weights=self.weights)
        (directory / "fold.json").write_text(json.dumps({
            "schema_version": 1, "period": float(self.period), "epoch": float(self.epoch),
            "fold_period": 2 * float(self.period), "local_windows": False,
            "model": "fold native exposure predictions with the data weights",
            "covariance": "sum_s A_s C_s A_s.T",
        }, indent=2) + "\n")

    @classmethod
    def load(cls, directory, *, mmap_mode="r"):
        directory = Path(directory)
        metadata = json.loads((directory / "fold.json").read_text())
        if metadata["schema_version"] != 1:
            raise ValueError("Unsupported folded-input schema")
        with np.load(directory / "fold_map.npz", allow_pickle=False) as arrays:
            return cls(load_block(directory / "block", mmap_mode=mmap_mode),
                       arrays["native_time"].copy(), arrays["bin_index"].copy(),
                       arrays["weights"].copy(), metadata["period"], metadata["epoch"])


def prepare_folded_fourier(blocks, *, period, epoch, bin_days, spectra=None):
    """Fold all observed phases at 2P, retaining gaps and alternating eclipses.

    Blocks describe independent sectors with one common exposure duration.
    Prefer the matching ``spectra`` returned by ``prepare_sector``: covariance
    is then propagated directly from each unchanged native PSD, as in the HZ
    production run. Without spectra, invert each block's positive-definite
    precision. Blocks with projected-out modes need the separate historical
    folded-covariance interface; a singular precision cannot be inverted here.

    White noise is already in the input covariance. This adds no noise floor,
    synthetic observations, eclipse windows, or discarded Fourier modes.
    """
    blocks = tuple(blocks)
    if not blocks or any(not isinstance(b, GaussianBlock) for b in blocks):
        raise ValueError("Require one or more GaussianBlock objects")
    if not np.allclose([b.exptime for b in blocks], blocks[0].exptime, rtol=1e-12, atol=0):
        raise ValueError("Folded models require a common native exposure duration")
    if spectra is not None:
        spectra = tuple(spectra)
        if len(spectra) != len(blocks):
            raise ValueError("Supply exactly one matching spectrum per block")
    native_time = np.concatenate([b.time for b in blocks])
    native_flux = np.concatenate([b.flux for b in blocks])
    operator, centers, bins, weights = full_orbit_fold(native_time, period, epoch, bin_days)
    covariance = np.zeros((len(centers), len(centers)))
    start = 0
    for index, block in enumerate(blocks):
        stop = start + len(block.time)
        ids, w = bins[start:stop], weights[start:stop]
        if spectra is None:
            native_cov = solve(block.precision, np.eye(len(block.time)), assume_a="pos")
        else:
            spectrum = spectra[index]
            grid, mask = np.asarray(spectrum["time"]), np.asarray(spectrum["observed"])
            if (grid.ndim != 1 or len(grid) < 2 or mask.dtype != bool
                    or mask.shape != grid.shape or not np.isfinite(grid).all()
                    or np.diff(grid)[0] <= 0
                    or not np.allclose(np.diff(grid), np.diff(grid)[0], rtol=1e-8, atol=1e-10)):
                raise ValueError("Require a regular PSD grid and matching observed mask")
            if grid[mask].shape != block.time.shape or not np.allclose(
                    grid[mask], block.time, rtol=0, atol=1e-8):
                raise ValueError("Spectrum observations and block ordering differ")
            native_cov = observed_covariance(spectrum["total_fft_power"], len(grid),
                                             np.flatnonzero(mask))
        native_cov *= w[:, None] * w[None, :]
        if len(np.unique(ids)) == len(ids):
            covariance[np.ix_(ids, ids)] += native_cov
        else:
            np.add.at(covariance, (ids[:, None], ids[None, :]), native_cov)
        start = stop
    flux = 1 + np.asarray(operator @ (native_flux - 1)).ravel()
    block = GaussianBlock.from_covariance(centers, flux, covariance, blocks[0].exptime)
    return FoldedFourierData(block, native_time, bins, weights, float(period), float(epoch))
