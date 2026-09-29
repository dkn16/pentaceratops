import os
import traceback
from astropy.coordinates import SkyCoord
from astropy import constants
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import astropy.units as u
import numpy as np
from scipy.integrate import dblquad
import pandas as pd
from pandas import DataFrame, read_csv
from math import floor, ceil
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from pytransit import QuadraticModel
from mpl_toolkits.axes_grid1.anchored_artists import (
    AnchoredDirectionArrows
    )

from .likelihoods.real import (simulate_TP_transit,
                         simulate_EB_transit,
                         simulate_EB_transit_secondary,
                         simulate_EB_transit_evenodd,
                         lnL_EB_second,
                         mean_anomaly_difference)
from .stellar import (Gauss2D,
                   save_trilegal,
                   query_TRILEGAL,
                   download_tesscut,
                   renorm_flux,
                   stellar_relations,
                   get_aperture)
from .evidence.real import *
from .sampling.policy import POLICY_VERSION, sampling_policy

from .evidence.eclipses import lnZ_TEB_secondary,lnZ_BEB_secondary,lnZ_DEB_secondary, lnZ_PEB_secondary,lnZ_SEB_secondary

from .evidence.eclipses import lnZ_TEB_evenodd,lnZ_SEB_evenodd,lnZ_PEB_evenodd,lnZ_DEB_evenodd,lnZ_BEB_evenodd

Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value
G = constants.G.cgs.value
au = constants.au.cgs.value
pi = np.pi
ln2pi = np.log(2*pi)


_secondary_nondetection_model = QuadraticModel(interpolate=False)


def _error_vector(error, size, label):
    """Return a positive per-point uncertainty vector.

    ``calc_probs`` historically accepts a scalar folded-light-curve error,
    but accepting a matching vector here makes the even/odd concatenation
    lossless when the two windows have different uncertainties.
    """
    values = np.asarray(error, dtype=float)
    if values.ndim == 0 or values.size == 1:
        values = np.full(int(size), float(values.reshape(-1)[0]))
    else:
        values = values.reshape(-1).copy()
        if values.size != int(size):
            raise ValueError(
                f"{label} uncertainty has length {values.size}; expected "
                f"{size}."
            )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{label} uncertainties must be finite and positive.")
    return values


def _concatenate_even_odd_observations(
        time_even, flux_even, err_even, time_odd, flux_odd, err_odd):
    """Build the common real-space dataset used by every scenario.

    The x2P EB likelihoods retain the two windows separately so that primary
    and secondary eclipse shapes may differ.  Every non-x2P likelihood sees
    these same observations concatenated, with a per-point uncertainty that
    remembers whether the datum came from the even or odd fold.  No padding
    or artificial unit-flux samples are introduced.
    """
    time_even = np.asarray(time_even, dtype=float).reshape(-1)
    flux_even = np.asarray(flux_even, dtype=float).reshape(-1)
    time_odd = np.asarray(time_odd, dtype=float).reshape(-1)
    flux_odd = np.asarray(flux_odd, dtype=float).reshape(-1)
    if time_even.size != flux_even.size:
        raise ValueError("even time and flux arrays must have equal length.")
    if time_odd.size != flux_odd.size:
        raise ValueError("odd time and flux arrays must have equal length.")

    sigma_even = _error_vector(err_even, time_even.size, "even")
    sigma_odd = _error_vector(err_odd, time_odd.size, "odd")
    # Give the ordinary (non-x2P) hypotheses equal numbers of even and odd
    # observations.  The original arrays remain untouched and are passed in
    # full to the x2P likelihoods.
    common_size = min(time_even.size, time_odd.size)
    if common_size == 0:
        raise ValueError("even and odd windows must not be empty.")
    time_even = time_even[:common_size]
    flux_even = flux_even[:common_size]
    sigma_even = sigma_even[:common_size]
    time_odd = time_odd[:common_size]
    flux_odd = flux_odd[:common_size]
    sigma_odd = sigma_odd[:common_size]
    valid_even = (
        np.isfinite(time_even) & np.isfinite(flux_even)
        & np.isfinite(sigma_even)
    )
    valid_odd = (
        np.isfinite(time_odd) & np.isfinite(flux_odd)
        & np.isfinite(sigma_odd)
    )
    if not np.any(valid_even) or not np.any(valid_odd):
        raise ValueError("even and odd windows must each contain valid data.")

    combined_time = np.concatenate(
        [time_even[valid_even], time_odd[valid_odd]]
    )
    combined_flux = np.concatenate(
        [flux_even[valid_even], flux_odd[valid_odd]]
    )
    combined_sigma = np.concatenate(
        [sigma_even[valid_even], sigma_odd[valid_odd]]
    )
    order = np.argsort(combined_time, kind="stable")
    return (
        combined_time[order], combined_flux[order], combined_sigma[order]
    )


def _centered_secondary_model_new(
        time_secondary, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2,
        ecc, argp, companion_fluxratio=0.0, companion_is_host=False,
        exptime=0.00139, nsamples=20):
    """Secondary template centred on a same-length zero-residual window.

    This helper belongs to the development ``triceratops_new`` target only.
    The intermediate ``triceratops`` snapshot and ``triceratops_original``
    remain unchanged.
    """
    time_secondary = np.asarray(time_secondary, dtype=float)
    _secondary_nondetection_model.set_data(
        time_secondary, exptimes=exptime, nsamples=nsamples
    )
    k = R_EB / R_s
    if abs(k - 1.0) < 1e-6:
        k *= 0.999
    sec_flux = _secondary_nondetection_model.evaluate_ps(
        k=1.0 / k,
        ldc=[float(u1), float(u2)],
        t0=0.0,
        p=P_orb,
        a=a / (k * R_s * Rsun),
        i=inc * (pi / 180.0),
        e=ecc,
        w=(90.0 - argp + 180.0) * (pi / 180.0),
    )

    F_target = 1.0
    F_comp = companion_fluxratio / (1.0 - companion_fluxratio)
    F_EB = EB_fluxratio / (1.0 - EB_fluxratio)
    if companion_is_host:
        sec_flux = (sec_flux + F_comp / F_EB) / (1.0 + F_comp / F_EB)
        F_dilute = F_target / (F_comp + F_EB)
    else:
        sec_flux = (sec_flux + F_target / F_EB) / (1.0 + F_target / F_EB)
        F_dilute = F_comp / (F_target + F_EB)
    return (sec_flux + F_dilute) / (1.0 + F_dilute)


def _lnL_EB_second_no_external_peak(
        time, flux, sigma, time_secondary, flux_secondary, sigma_secondary,
        R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp,
        companion_fluxratio=0.0, companion_is_host=False,
        exptime=0.00139, nsamples=20):
    """New-model EB likelihood with a zero-residual out-of-window fallback.

    When the physical secondary centre is inside the supplied window, this is
    exactly the existing joint primary/secondary likelihood.  When it is
    outside, retain the observed window's flat-model cost and add the
    null-referenced cost of the centred secondary template against a
    same-length zero-residual light curve.  This encodes the explicit
    assumption that no secondary peak exists elsewhere without changing the
    original implementation.
    """
    time_secondary = np.asarray(time_secondary, dtype=float)
    flux_secondary = np.asarray(flux_secondary, dtype=float)
    secondary_offset = (
        mean_anomaly_difference(ecc, argp * (pi / 180.0)) - 0.5
    ) * P_orb
    if np.min(time_secondary) <= secondary_offset <= np.max(time_secondary):
        return lnL_EB_second(
            time, flux, sigma,
            time_secondary, flux_secondary, sigma_secondary,
            R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp,
            companion_fluxratio, companion_is_host, exptime, nsamples,
        )

    primary_model, _ = simulate_EB_transit_secondary(
        time, time_secondary,
        R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp,
        companion_fluxratio, companion_is_host, exptime, nsamples,
    )
    centered_secondary = _centered_secondary_model_new(
        time_secondary, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2,
        ecc, argp, companion_fluxratio, companion_is_host, exptime, nsamples,
    )
    primary_cost = 0.5 * np.sum((flux - primary_model)**2 / sigma**2)
    observed_flat_cost = 0.5 * np.sum(
        (flux_secondary - 1.0)**2 / sigma_secondary**2
    )
    zero_residual_cost = 0.5 * np.sum(
        (1.0 - centered_secondary)**2 / sigma_secondary**2
    )
    return primary_cost + observed_flat_cost + zero_residual_cost


class target:
    def __init__(self, ID: int, sectors: np.ndarray,
                 search_radius: int = 10, mission: str = "TESS",
                 lightkurve_cache_dir = None, trilegal_fname = None,
                 ra: float=None, dec: float=None, verify_ssl: bool=True,
                 mag_lim: str = "21"):
        """Initializes TRICERATOPS.

        Queries TIC for sources near the target and obtains a cutout
        of the pixels enclosing the target.

        Args:
            ID (int): TIC ID of the target.
            sectors (numpy array): Sectors in which the target
                has been observed. If Kepler or K2 selected, sectors
                corresponds to quarter or campaign, respectively.
            search_radius (int): Number of pixels from the target
                star to search.
            mission (str): "TESS", "Kepler", or "K2".
            lightkurve_cache_dir (str): Path to lightkurve cache
                directory.
            trilegal_fname (str): Path to trilegal table.
            ra (float): right ascension of target.
            dec (float): declination of target.
            verify_ssl (bool): True to verify SSL certificates,
                False to ignore. ONLY SET TO FALSE IF ABSOLUTELY
                NECESSARY.
            mag_lim (str): Limiting magnitude for TRILEGAL simulation
                (default "21").
        """
        # Catalog access is optional; importing numerical modules is offline.
        try:
            import lightkurve
            from astroquery.mast import Catalogs, Tesscut
            from astroquery.vizier import Vizier
        except ImportError as exc:
            raise ImportError("Target catalog queries require pentaceratops[catalogs]") from exc
        self.ID = ID
        self.mission = mission
        if mission != "TESS" and mission != "Kepler" and mission != "K2":
            raise ValueError("Introduced invalid mission: " + mission)
        self.mission = mission
        self.sectors = sectors
        self.search_radius = search_radius
        self.N_pix = 2*search_radius+2
        # query TIC for nearby stars
        if mission == "TESS":
            pixel_size = 20.25*u.arcsec
        else:
            pixel_size = 4*u.arcsec
        if mission == 'TESS':
            ticid = ID
        else:
            if ra is None or dec is None:
                if mission == "Kepler":
                    columns = ["_RA", "_DE"]
                    result = (
                        Vizier(columns=columns)
                            .query_constraints(
                                KIC=str(ID),
                                catalog="J/ApJS/229/30/catalog"
                                )[0].as_array()
                    )
                    ra = result[0]["_RA"]
                    dec = result[0]["_DE"]
                elif mission == "K2":
                    result = (
                        Vizier(columns=["RAJ2000", "DEJ2000"])
                            .query_constraints(
                                ID=str(ID),
                                catalog="IV/34/epic"
                                )[0].as_array()
                    )
                    ra = result[0]["RAJ2000"]
                    dec = result[0]["DEJ2000"]
            ticid = Catalogs.query_region(
                SkyCoord(ra, dec, unit="deg"),
                radius=search_radius * pixel_size,
                catalog="TIC"
            )[0]["ID"]
        df = Catalogs.query_object(
            "TIC"+str(ticid),
            radius=search_radius*pixel_size,
            catalog="TIC"
            )
        new_df = df[
            "ID", "Tmag", "Jmag", "Hmag", "Kmag",
            "ra", "dec", "mass", "rad", "Teff", "plx",
            # for spurious objects, etc.
            # see https://arxiv.org/abs/2108.04778 Section 3.1 for details
            "disposition", "duplicate_id"
            ]
        stars = new_df.to_pandas()

        # start TRILEGAL query if needed
        if trilegal_fname is None:
            output_url = query_TRILEGAL(
                stars["ra"].values[0],
                stars["dec"].values[0],
                verbose=0,
                verify_ssl=verify_ssl,
                mag_lim=mag_lim
                )
            self.trilegal_url = output_url
            self.trilegal_fname = None
        else:
            self.trilegal_fname = trilegal_fname
            self.trilegal_url = None

        TESS_images = []
        col0s, row0s = [], []
        pix_coords = []
        Tmag = stars["Tmag"].values
        ra = stars["ra"].values
        dec = stars["dec"].values
        cutout_coord = SkyCoord(ra[0], dec[0], unit="deg")
        # for each sector, get FFI cutout and transform RA/Dec into
        # TESS pixel coordinates
        for j, sector in enumerate(sectors):
            try:
                if mission == "TESS":
                    print(f"Getting TessCut for sector {sector}")
                    tess_cuts = download_tesscut(
                        lightkurve.search_tesscut(
                            target=cutout_coord, sector=sector
                        ),
                        cutout_size=(self.N_pix, self.N_pix),
                        download_dir=lightkurve_cache_dir,
                    )
                    cutout_hdu = tess_cuts[0].hdu
                    cutout_table = cutout_hdu[1].data
                    hdu = cutout_hdu[2].header
                    wcs = WCS(hdu)
                    n_cols_before = 0
                    n_rows_before = 0
                    TESS_images.append(np.nanmean(cutout_table["FLUX"], axis=0))
                    col0 = cutout_hdu[1].header["1CRV4P"]
                    row0 = cutout_hdu[1].header["2CRV4P"]
                elif mission == "Kepler":
                    print(f"Getting TPF for sector {sector}")
                    tpf = lightkurve.search_targetpixelfile(
                        "KIC " + str(ID),
                        mission="Kepler",
                        quarter=sector
                        ).download_all(download_dir=lightkurve_cache_dir)
                    cutout_table = tpf[0].hdu[1].data
                    hdu = tpf[0].hdu[2].header
                    wcs = WCS(hdu)
                    image = np.nanmean(cutout_table["FLUX"], axis=0)
                    n_rows_before = (self.N_pix - image.shape[0])//2
                    n_rows_after = ((self.N_pix - image.shape[0])
                        - (self.N_pix - image.shape[0])//2)
                    n_cols_before = (self.N_pix - image.shape[1])//2
                    n_cols_after = ((self.N_pix - image.shape[1])
                        - (self.N_pix - image.shape[1])//2)
                    npad = ((n_rows_before, n_rows_after),
                            (n_cols_before, n_cols_after))
                    image = np.pad(
                        image, npad, mode='constant', constant_values=(np.nan)
                        )
                    TESS_images.append(image)
                    col0 = tpf[0].hdu[1].header["1CRV4P"] - n_cols_before
                    row0 = tpf[0].hdu[1].header["2CRV4P"] - n_rows_before
                elif mission == "K2":
                    print(f"Getting TPF for sector {sector}")
                    tpf = lightkurve.search_targetpixelfile(
                        "EPIC " + str(ID),
                        mission="K2",
                        campaign=sector
                        ).download_all(download_dir=lightkurve_cache_dir)
                    cutout_table = tpf[0].hdu[1].data
                    hdu = tpf[0].hdu[2].header
                    wcs = WCS(hdu)
                    image = np.nanmean(cutout_table["FLUX"], axis=0)
                    n_rows_before = (self.N_pix - image.shape[0])//2
                    n_rows_after = ((self.N_pix - image.shape[0])
                        - (self.N_pix - image.shape[0])//2)
                    n_cols_before = (self.N_pix - image.shape[1])//2
                    n_cols_after = ((self.N_pix - image.shape[1])
                        - (self.N_pix - image.shape[1])//2)
                    npad = ((n_rows_before, n_rows_after),
                            (n_cols_before, n_cols_after))
                    image = np.pad(
                        image, npad, mode='constant', constant_values=(np.nan)
                        )
                    TESS_images.append(image)
                    col0 = tpf[0].hdu[1].header["1CRV4P"] - n_cols_before
                    row0 = tpf[0].hdu[1].header["2CRV4P"] - n_rows_before
            except Exception as e:
                print(f"Sector {sector} raised exception. Ignoring for validation.")
                print(traceback.format_exc())
                continue
            col0s.append(col0)
            row0s.append(row0)

            pix_coord = np.zeros([len(ra), 2])
            for i in range(len(ra)):
                RApix = wcs.all_world2pix(ra[i], dec[i], 0)[0].item()
                Decpix = wcs.all_world2pix(ra[i], dec[i], 0)[1].item()
                pix_coord[i, 0] = col0+RApix + n_cols_before
                pix_coord[i, 1] = row0+Decpix + n_rows_before
            pix_coords.append(pix_coord)

        # for each star, get the separation and position angle
        # from the targets star
        sep = [0]
        pa = [0]
        c_target = SkyCoord(
            stars["ra"].values[0],
            stars["dec"].values[0],
            unit="deg"
            )
        for i in range(1,len(stars)):
            c_star = SkyCoord(
                stars["ra"].values[i],
                stars["dec"].values[i],
                unit="deg"
                )
            sep.append(
                np.round(
                    c_target.separation(c_star).to(u.arcsec).value,
                    3
                    )
                )
            pa.append(
                np.round(
                    c_target.position_angle(c_star).to(u.deg).value,
                    3
                    )
                )
        stars["sep (arcsec)"] = sep
        stars["PA (E of N)"] = pa

        self.stars = stars
        self.TESS_images = TESS_images
        self.col0s = col0s
        self.row0s = row0s
        self.pix_coords = pix_coords
        return

    def add_star(self, ID: int, Tmag: float, bound: bool):
        """For adding newly identified stars.

        Adds an additional star (e.g., an unresolved star identified
        with follow up) to the .stars dataframe.

        Args:
            ID (int): Arbitrary ID for the new star.
            Tmag (float): Estimated TESS magnitude of the new star.
            bound (bool): True if new star is physically bound to
                target star, False if new star is unbound.
        """
        # if bound, use same parallax as target star
        if bound:
            plx = self.stars['plx'].values[0]
            new_star = DataFrame(
                [[str(ID), Tmag, plx]], columns=["ID", "Tmag", "plx"]
                )
        else:
            new_star = DataFrame(
                [[str(ID), Tmag]], columns=["ID", "Tmag"]
                )
        # MUST reset_index of the concatenated data frame
        # to ensure the indices of each row is unique.
        # Otherwise update_star() might produce incorrect result
        # (updating multiple rows with the same index)
        self.stars = pd.concat([self.stars, new_star]).reset_index(drop=True)

        # for each set of pixel coordinates (corresponding to
        # each sector), append a row for the new star with the same
        # coordinates as the target star
        for i in range(len(self.pix_coords)):
            self.pix_coords[i] = np.append(
                self.pix_coords[i],
                self.pix_coords[i][0]
                ).reshape(
                len(self.pix_coords[i])+1, 2
                )
        return

    def remove_star(self, drop_stars: np.ndarray):
        """For removing stars ruled out from being NTPs or NEBs.

        Drops stars from .stars dataframe so that they are
        excluded from validation analysis.

        Args:
            drop_stars (numpy array): Array of (int) TIC IDs
                for stars to drop.
        """
        # convert drop_stars to an array of str for actual matching
        if np.isscalar(drop_stars):
            drop_stars = [drop_stars]
        drop_stars = [str(s) for s in drop_stars]
        self.stars = self.stars[~self.stars["ID"].isin(drop_stars)]
        return

    def update_star(self, ID: int, param: str, value: float):
        """For updating the properties of a star.

        Updates parameters of a star in .stars dataframe.

        Args:
            ID (int): ID of star to edit.
            param (str): Name of parameter to edit
                         (i.e., the header of .stars dataframe)
            value (float): Value to update the parameter to.
        """
        idx = self.stars[self.stars.ID == str(ID)].index
        self.stars.loc[idx, [param]] = value
        return

    def _infer_stellar_age(self, default_age_gyr: float = 5.0) -> float:
        """Try to pull a FLAME age (Gyr) for the target from Gaia DR3
        astrophysical_parameters. Fall back to `default_age_gyr` (main-sequence)
        if unavailable or non-finite.
        """
        from .companions.load import get_target_gaia_age
        ra = float(self.stars["ra"].values[0])
        dec = float(self.stars["dec"].values[0])
        age = get_target_gaia_age(ra, dec)
        if age is None:
            print(
                f"Gaia FLAME age unavailable for target; defaulting to "
                f"{default_age_gyr} Gyr (main-sequence). Pass an explicit "
                f"star_age to run_stecomp() for young / evolved targets."
            )
            return float(default_age_gyr)
        return float(age)

    def run_stecomp(self, star_age: float = None, n_sim: int = 1_000_000,
                    out_path: str = None, force: bool = False) -> str:
        """Run the stecomp (MOLUSC rewrite) companion simulator for this
        target. Returns a path to a CSV whose schema matches the
        `molusc_file` contract consumed by the STP/SEB scenarios, so the
        result can be passed straight into calc_probs(molusc_file=...).

        Args:
            star_age: Stellar age in Gyr. If None, attempts to pull from Gaia
                DR3 FLAME ages; falls back to 5.0 Gyr (main-sequence) with a
                warning when unavailable.
            n_sim: Number of Monte Carlo companion draws.
            out_path: Destination CSV path. Defaults to
                "<TIC ID>_stecomp.csv" in the current directory.
            force: If True, re-run even when `out_path` already exists.

        Returns:
            Path to the CSV (cached or freshly written).
        """
        if out_path is None:
            out_path = f"{self.ID}_stecomp.csv"

        if not force and os.path.exists(out_path):
            if getattr(self, "_stecomp_out_path", None) != out_path:
                print(f"Reusing existing stecomp CSV at {out_path}")
            self._stecomp_out_path = out_path
            return out_path

        from .companions.sample import stellar_companion

        if star_age is None:
            star_age = self._infer_stellar_age()

        ra = float(self.stars["ra"].values[0])
        dec = float(self.stars["dec"].values[0])
        primary_mass = float(self.stars["mass"].values[0])

        try:
            pop = stellar_companion(ra, dec, primary_mass, star_age, n_sim=n_sim)
        except Exception as exc:
            print(f"stecomp failed ({exc}); falling back to analytic priors.")
            self._stecomp_out_path = None
            return None

        # Write in the schema the existing molusc_file consumer expects.
        df = pd.DataFrame({
            "mass ratio": pop["mass_ratios"],
            "period(days)": pop["periods"],
            "semi-major axis(AU)": pop["semi_major_axes"],
            "cos_i": np.cos(pop["inclinations"]),
            "eccentricity": pop["eccentricities"],
            "arg periastron": pop["arguments_of_periapsis"],
            "phase": pop["pericenter_phase"],
            "Projected Separation(AU)": pop["proj_sep_au"],
            "DeltaG": pop["delta_g"],
            "Predicted RUWE": pop["predicted_ruwe"],
        })
        df.to_csv(out_path, index=False)
        print(
            f"stecomp: {pop['n_accepted']}/{pop['n_total']} companions survived; "
            f"wrote {out_path}"
        )
        self._stecomp_out_path = out_path
        return out_path

    def get_spoc_apertures(self):
        """
        Returns apertures used by the SPOC in the given
        sectors, if available.
        Args:
            self
        Returns:
            aps (list): List of aperture pixels, in order of
                        sectors as input.
        """
        aps = []
        this_ID = self.ID
        these_sectors = self.sectors
        try:
            for sector in these_sectors:
                ap = get_aperture(this_ID, sector)
                aps.append(ap)
        except:
            print("No SPOC apertures available.")
        return aps

    def plot_field(self, sector: int = None, ap_pixels = None,
                   ap_color: str = "red", save: bool = False,
                   fname: str = None):
        """For visualizing the field of stars.

        Plots the field of stars and pixels around the target to
        show their positions relative to the TESS pixels.

        Args:
            sector (int): Sector to plot.
            ap_pixels (numpy array): Aperture used to
                extract light curve.
            ap_color (str): Color of aperture outline.
            save (bool): Whether or not to save plot as pdf.
            fname (str): File name of pdf.
        """
        if len(self.sectors) > 1:
            idx = np.argwhere(self.sectors == sector)[0, 0]
        else:
            idx = 0

        corners = np.arange(-0.5, self.N_pix+0.5, 1)
        centers = np.arange(0, self.N_pix, 1)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
        plt.subplots_adjust(right=0.9)
        # aperture
        if ap_pixels is not None:
            for i in range(len(ap_pixels)):
                ax[0].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]-0.5],
                    color=ap_color, zorder=1
                    )
                ax[0].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]+0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=1
                    )
                ax[0].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]-0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=1
                    )
                ax[0].plot(
                    [ap_pixels[i][0]+0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=1
                    )
        # pixel grid
        for i in corners:
            ax[0].plot(
                np.full_like(corners, self.col0s[idx]+i),
                self.row0s[idx]+corners, "k-", lw=0.5,
                zorder=0
                )
            ax[0].plot(
                self.col0s[idx]+corners,
                np.full_like(corners, self.row0s[idx]+i), "k-", lw=0.5,
                zorder=0
                )
        # search radius
        ax[0].plot(
            (
                self.pix_coords[idx][0, 0]
                + self.search_radius
                * np.cos(np.linspace(0, 2*pi, 100))
            ),
            (
                self.pix_coords[idx][0, 1]
                + self.search_radius
                * np.sin(np.linspace(0, 2*pi, 100))
            ),
            "k--", alpha=0.5, zorder=0)
        # N and E arrows
        if len(self.pix_coords[idx]) > 1:
            v1 = np.array([0, 1])
            v2 = self.pix_coords[idx][1] - self.pix_coords[idx][0]
            sign = np.sign(v2[0])
            angle1 = sign * (
                np.arccos(np.dot(v1, v2)
                / np.sqrt((np.dot(v1, v1)) * np.dot(v2, v2))) * 180/np.pi
                )
            angle2 = self.stars["PA (E of N)"].values[1]
            rot = angle1-angle2
            rotated_arrow = AnchoredDirectionArrows(
                                ax[0].transAxes,
                                "E", "N",
                                loc="upper left",
                                color="k",
                                angle=-rot,
                                length=0.1,
                                fontsize=0.05,
                                back_length=0,
                                head_length=5,
                                head_width=5,
                                tail_width=1
                                )
            ax[0].add_artist(rotated_arrow)
        # stars
        sc = ax[0].scatter(
            self.pix_coords[idx][1:, 0],
            self.pix_coords[idx][1:, 1],
            c=self.stars["Tmag"].values[1:], s=75,
            edgecolors="k",
            cmap=cm.viridis_r,
            vmin=floor(min(self.stars["Tmag"])),
            vmax=ceil(max(self.stars["Tmag"])),
            zorder=2,
            rasterized=True
            )
        ax[0].scatter(
            [self.pix_coords[idx][0, 0]],
            [self.pix_coords[idx][0, 1]],
            c=[self.stars["Tmag"].values[0]], s=250,
            marker="*",
            edgecolors="k",
            cmap=cm.viridis_r,
            vmin=floor(min(self.stars["Tmag"])),
            vmax=ceil(max(self.stars["Tmag"])),
            zorder=2
            )
        cb1 = fig.colorbar(sc, ax=ax[0], pad=0.02)
        cb1.ax.set_ylabel(
            "TESS mag", rotation=270, fontsize=12, labelpad=18
            )
        ax[0].set_ylim([
            min(self.row0s[idx]+corners),
            max(self.row0s[idx]+corners)
            ])
        ax[0].set_xlim([
            min(self.col0s[idx]+corners),
            max(self.col0s[idx]+corners)
            ])
        ax[0].set_yticks(self.row0s[idx]+centers)
        ax[0].set_xticks(self.col0s[idx]+centers)
        ax[0].tick_params(width=0)
        ax[0].tick_params(axis='x', labelrotation=90)
        ax[0].set_ylabel("pixel row number", fontsize=12)
        ax[0].set_xlabel("pixel column number", fontsize=12)
        # TESS FFI
        im = ax[1].imshow(
            self.TESS_images[idx],
            extent=[
                min(self.col0s[idx]+corners),
                max(self.col0s[idx]+corners),
                max(self.row0s[idx]+corners),
                min(self.row0s[idx]+corners)
            ])
        cb2 = fig.colorbar(im, ax=ax[1], pad=0.02)
        cb2.ax.set_ylabel(
            "flux [e$^-$ s$^{-1}$]",
            rotation=270, fontsize=12, labelpad=18)
        ax[1].set_ylim([
            min(self.row0s[idx]+corners),
            max(self.row0s[idx]+corners)
            ])
        ax[1].set_xlim([
            min(self.col0s[idx]+corners),
            max(self.col0s[idx]+corners)
            ])
        ax[1].set_yticks(self.row0s[idx]+centers)
        ax[1].set_xticks(self.col0s[idx]+centers)
        ax[1].tick_params(width=0)
        ax[1].tick_params(axis='x', labelrotation=90)
        ax[1].set_ylabel("pixel row number", fontsize=12)
        ax[1].set_xlabel("pixel column number", fontsize=12)
        # aperture
        if ap_pixels is not None:
            for i in range(len(ap_pixels)):
                ax[1].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]-0.5],
                    color=ap_color, zorder=2
                    )
                ax[1].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]+0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=2
                    )
                ax[1].plot(
                    [ap_pixels[i][0]-0.5, ap_pixels[i][0]-0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=2
                    )
                ax[1].plot(
                    [ap_pixels[i][0]+0.5, ap_pixels[i][0]+0.5],
                    [ap_pixels[i][1]-0.5, ap_pixels[i][1]+0.5],
                    color=ap_color, zorder=2
                    )
        if save is False:
            plt.tight_layout()
            plt.show()
        elif (save is True) & (fname is None):
            plt.tight_layout()
            target_star = self.stars.ID.values[0]
            plt.savefig("TIC"+str(target_star)+"_sector"+str(sector)+".pdf")
        else:
            plt.tight_layout()
            plt.savefig(fname+".pdf")
        return

    def calc_depths(self, tdepth: float, all_ap_pixels = None):
        """Calculates required transit depth of each star.

        Calculates the transit depth each source near the target would
        have if it were the source of the transit.
        This is done by modeling the PSF of each source as a circular
        Gaussian with a standard deviation of 0.75 pixels.

        Args:
            tdepth (float): Reported transit depth [ppm].
            all_ap_pixels (list of numpy arrays): Apertures used to
                extract light curve.
        """
        if all_ap_pixels is None:
            print("No apertures provided, assuming 5x5 centered on target.")
            all_ap_pixels = []
            for i in range(len(self.pix_coords)):
                target_pixel = np.round(self.pix_coords[i][0])
                this_ap = np.array([
                    np.repeat(
                        np.arange(
                            target_pixel[0] - 2, target_pixel[0] + 3, 1
                            ), 5
                        ),
                    np.tile(
                        np.arange(
                            target_pixel[1] - 2, target_pixel[1] + 3, 1
                            ), 5
                        )
                    ]).T
                all_ap_pixels.append(this_ap)
        # for each aperture, calculate contribution due to each star
        rel_flux_per_aperture = np.zeros([
            len(all_ap_pixels),
            len(self.stars)
            ])
        flux_ratio_per_aperture = np.zeros([
            len(all_ap_pixels),
            len(self.stars)
            ])
        for k in range(len(all_ap_pixels)):
            for i in range(len(self.stars)):
                # location of star in pixel space for aperture k
                mu_x = self.pix_coords[k][i, 0]
                mu_y = self.pix_coords[k][i, 1]
                # star's flux normalized to brightest star
                A = 10**((
                    np.min(self.stars.Tmag.values)
                    - self.stars.Tmag.values[i]
                    )/2.5)
                # integrate PSF in each pixel
                this_flux = 0
                for j in range(len(all_ap_pixels[k])):
                    this_pixel = all_ap_pixels[k][j]
                    this_flux += dblquad(
                        Gauss2D,
                        this_pixel[1]-0.5,
                        this_pixel[1]+0.5,
                        this_pixel[0]-0.5,
                        this_pixel[0]+0.5,
                        args=(mu_x, mu_y, 0.75, A))[0]
                rel_flux_per_aperture[k, i] = this_flux
            # calculate flux ratios for this aperture
            flux_ratio_per_aperture[k, :] = (
                rel_flux_per_aperture[k, :]
                / np.sum(rel_flux_per_aperture[k])
                )

        # take average of flux ratios across all apertures and append
        # to stars dataframe
        flux_ratios = np.mean(flux_ratio_per_aperture, axis=0)
        self.stars["fluxratio"] = flux_ratios
        # calculate transit depth of each star given input transit depth
        tdepths = np.zeros(len(self.stars))
        for i in range(len(flux_ratios)):
            if flux_ratios[i] != 0:
                tdepths[i] = 1-(flux_ratios[i]-tdepth)/flux_ratios[i]
        tdepths[tdepths > 1] = 0
        self.stars["tdepth"] = tdepths

        # check target and possible NFPs for missing properties
        filtered_stars = self.stars[self.stars["tdepth"] > 0]
        for i, ID in enumerate(filtered_stars["ID"].values):
            M_s = filtered_stars["mass"].values[i]
            R_s = filtered_stars["rad"].values[i]
            Teff = filtered_stars["Teff"].values[i]
            Tmag = filtered_stars["Tmag"].values[i]
            plx = filtered_stars["plx"].values[i]
            if i == 0:
                if (np.isnan(M_s) or np.isnan(R_s)
                    or np.isnan(Teff) or np.isnan(plx)):
                    print(
                        "WARNING: " + str(ID)
                        + " is missing stellar properties required "
                        + "for validation."
                        + "Please ensure a stellar "
                        + "mass (in M_Sun), radius (in R_Sun), "
                        + "Teff (in K), and plx (in mas) "
                        + "are provided in the .stars dataframe."
                        )
                    print('M_s{}, R_s{}, Teff{}, plx{}'.format(M_s, R_s, Teff, plx))
                    raise ValueError("Target star missing stellar properties required for validation.")
            else:
                if (np.isnan(M_s) or np.isnan(R_s)
                    or np.isnan(Teff)):
                    print(
                        "WARNING: " + str(ID)
                        + " is missing stellar properties. "
                        + "If a mass (in M_Sun), "
                        + "radius (in R_Sun), and/or Teff (in K) "
                        + "are not added to the .stars dataframe, "
                        + "Solar values will be assumed."
                        )
        return

    def _stash_posterior(self, j, scenarios, targets, res):
        """Store the full per-scenario posterior samples.

        Called from calc_probs only when output_posteriors=True. ``res`` is the
        dict returned by an lnZ_* function; its per-parameter entries are
        equal-weight posterior draws (count set by posterior_nsamples via the
        POSTERIOR_NSAMPLES module globals). The samples are stored in
        self.posteriors keyed by "<scenario>_<target ID>".
        """
        keys = ["M_s", "R_s", "u1", "u2", "P_orb", "inc", "b", "R_p",
                "ecc", "argp", "M_EB", "R_EB", "fluxratio_EB", "fluxratio_comp"]
        cols = {}
        for k in keys:
            if k in res:
                cols[k] = np.atleast_1d(np.asarray(res[k], dtype=float))
        if not cols:
            return
        n = max(len(v) for v in cols.values())
        for k in list(cols):
            if len(cols[k]) == 1 and n > 1:
                cols[k] = np.broadcast_to(cols[k], (n,)).copy()
        cols["lnZ"] = np.full(n, float(res.get("lnZ", np.nan)))
        label = "{}_{}".format(str(scenarios[j]), int(targets[j]))
        self.posteriors[label] = DataFrame(cols)

    def calc_probs(self, time: np.ndarray, flux_0: np.ndarray,
                   flux_err_0: float, P_orb,
                   contrast_curve_file: str = None, filt: str = "TESS",
                   N: int = None, steps: int = None,
                   drop_scenario: list = [],
                   verbose: int = 1, flatpriors: bool = False,
                   exptime: float = 0.00139, nsamples: int = 20,
                   molusc_file: str = None,
                   use_stecomp: bool = False,
                   stellar_age: float = None,
                   stecomp_n_sim: int = 1_000_000,
                   stecomp_out_path: str = None,
                   flux_secondary:np.ndarray = None,
                    time_secondary:np.ndarray = None,
                    err_secondary:float = None,
                    assume_no_secondary_outside: bool = True,
                    flux_even:np.ndarray = None,
                    time_even:np.ndarray = None,
                    err_even:float = None,
                    flux_odd:np.ndarray = None,
                    time_odd:np.ndarray = None,
                    err_odd:float = None,
                    output_posteriors: bool = False,
                    posterior_nsamples: int = 2000
    ):
        """Run to calculate FPP and NFPP.

        Calculates the relative probability of each scenario.

        Args:
            time (numpy array): Time of each data point
                [days from transit midpoint].
            flux_0 (numpy array): Normalized flux of each data point.
            flux_err_0 (float): Uncertainty of flux.
            P_orb (float or numpy array): Orbital period [days] OR
                min and max periods to consider (i.e., [P_min, P_max]).
            contrast_curve_file (str): Path to contrast curve text file.
                File should contain column with separations (in arcsec)
                followed by column with Delta_mags.
            filt (str): Photometric filter of contrast curve. Options are
                TESS, Vis, J, H, and K.
            N (int or None): Global particle override; None uses scenario defaults.
            steps (int or None): Global MCMC-step override; None uses scenario defaults.
            drop_scenario (list of strings): Scenarios to ignore
                (e.g., ["TEB", "PEB"]).
            verbose (int): 1 to print progress, 0 to print nothing.
            exptime (float): Exposure time of observations [days].
            nsamples (int): Sampling rate for supersampling.
            molusc_file (str): Path to MOLUSC output with stellar
                binary properties.
            use_stecomp (bool): If True and no molusc_file is given, run
                the vendored stecomp (MOLUSC rewrite) to produce a
                Gaia-constrained companion population and use it as
                molusc_file. Cached to disk; will not re-run if the CSV
                already exists.
            stellar_age (float): Stellar age in Gyr for stecomp. If None,
                inferred from Gaia DR3 FLAME ages when available, else
                defaults to 5 Gyr (main-sequence).
            stecomp_n_sim (int): Monte Carlo draws for stecomp.
            stecomp_out_path (str): Destination CSV path for the stecomp
                output. Defaults to "<TIC ID>_stecomp.csv".
            assume_no_secondary_outside (bool): For eccentric EB models whose
                predicted secondary-eclipse centre lies outside the supplied
                secondary window, compare a centred secondary template with a
                same-length zero-residual light curve. This affects
                ``triceratops_new`` only. Set to False to retain the previous
                secondary-window likelihood.
        """
        # Resolve before expensive setup; overrides apply independently per field.
        self.sampling_policy_version = POLICY_VERSION
        self.sampling_config = sampling_policy(N=N, steps=steps)

        def sampling_pair(scenario):
            config = self.sampling_config[scenario]
            return config["N"], config["steps"]

        def twin_sampling(scenario):
            config = self.sampling_config[scenario]
            return dict(N_twin=config["N"], steps_twin=config["steps"])

        # If caller requested stecomp and did not supply an explicit
        # molusc_file, run (or reuse) the stecomp CSV.
        if use_stecomp and molusc_file is None:
            molusc_file = self.run_stecomp(
                star_age=stellar_age,
                n_sim=stecomp_n_sim,
                out_path=stecomp_out_path,
            )
        secondary_loglike = (
            _lnL_EB_second_no_external_peak
            if assume_no_secondary_outside else lnL_EB_second
        )
        # remove nans from light curve
        mask = ~np.isnan(time) & ~np.isnan(flux_0)
        time = time[mask]
        flux_0 = flux_0[mask]
        # Keep immutable references to every caller-supplied window.  The
        # per-star loop below renormalizes each window into that star's flux
        # frame.  Reusing those loop-local arrays on the next iteration
        # compounds the dilution correction
        # across stars (catastrophically for faint nearby stars).
        time_primary_base = time
        time_secondary_base = time_secondary
        flux_secondary_base = flux_secondary
        err_secondary_base = err_secondary
        time_even_base = time_even
        flux_even_base = flux_even
        err_even_base = err_even
        time_odd_base = time_odd
        flux_odd_base = flux_odd
        err_odd_base = err_odd
        even_odd_values = (
            time_even_base, flux_even_base, err_even_base,
            time_odd_base, flux_odd_base, err_odd_base,
        )
        has_any_even_odd = any(value is not None for value in even_odd_values)
        has_even_odd = all(value is not None for value in even_odd_values)
        if has_any_even_odd and not has_even_odd:
            raise ValueError(
                "even/odd likelihood data are incomplete: time, flux, and "
                "uncertainty must be supplied for both folds."
            )
        # Precompute secondary lnL correction (0 when no secondary data)
        if flux_secondary is not None and err_secondary is not None:
            _sec_lnL_correction = np.sum((flux_secondary - 1)**2 / (2 * err_secondary**2))
        else:
            _sec_lnL_correction = 0.0
        # construct a new dataframe that gives the values of lnL, best
        # fit parameters, lnprior, and relative probability of
        # each scenario considered
        # Posterior export: when output_posteriors=True, raise the number of
        # equal-weight posterior draws each lnZ_* retains, and collect every
        # scenario's posterior into self.posteriors. Default (False) restores
        # POSTERIOR_NSAMPLES=100 so the best-fit-only behaviour is unchanged.
        from .evidence import real as _mln_new
        from .evidence import eclipses as _mln_eb
        _post_n = int(posterior_nsamples) if output_posteriors else 100
        _mln_new.POSTERIOR_NSAMPLES = _post_n
        _mln_eb.POSTERIOR_NSAMPLES = _post_n
        self.posteriors = {}

        filtered_stars = self.stars[self.stars["tdepth"] > 0]
        N_scenarios = 3*len(filtered_stars) + 12
        targets = np.zeros(N_scenarios, dtype=np.dtype("i8"))
        star_num = np.zeros(N_scenarios, dtype=np.dtype("i8"))
        scenarios = np.zeros(N_scenarios, dtype=np.dtype('U6'))
        best_M_host = np.zeros(N_scenarios)
        best_R_host = np.zeros(N_scenarios)
        best_u1 = np.zeros(N_scenarios)
        best_u2 = np.zeros(N_scenarios)
        best_P_orb = np.zeros(N_scenarios)
        best_i = np.zeros(N_scenarios)
        best_b = np.zeros(N_scenarios)
        best_R_p = np.zeros(N_scenarios)
        best_ecc = np.zeros(N_scenarios)
        best_argp = np.zeros(N_scenarios)
        best_M_EB = np.zeros(N_scenarios)
        best_R_EB = np.zeros(N_scenarios)
        best_fluxratio_EB = np.zeros(N_scenarios)
        best_fluxratio_comp = np.zeros(N_scenarios)
        lnZ = np.zeros(N_scenarios)
        best_lnL = np.full(N_scenarios, np.nan)

        for i, ID in enumerate(filtered_stars["ID"].values):
            # Reset optional windows before every star.  renorm_flux returns
            # new arrays, so each stellar-frame correction must remain local
            # to this iteration.
            time = time_primary_base
            time_secondary = time_secondary_base
            flux_secondary = flux_secondary_base
            err_secondary = err_secondary_base
            time_even = time_even_base
            flux_even = flux_even_base
            err_even = err_even_base
            time_odd = time_odd_base
            flux_odd = flux_odd_base
            err_odd = err_odd_base
            # subtract flux from other stars in the aperture
            flux, flux_err = renorm_flux(
                flux_0, flux_err_0, filtered_stars["fluxratio"].values[i]
                )
            if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                flux_secondary, err_secondary = renorm_flux(
                    flux_secondary, err_secondary, filtered_stars["fluxratio"].values[i]
                    )
            if flux_even is not None and time_even is not None and err_even is not None:
                flux_even, err_even = renorm_flux(
                    flux_even, err_even, filtered_stars["fluxratio"].values[i]
                    )
            if flux_odd is not None and time_odd is not None and err_odd is not None:
                flux_odd, err_odd = renorm_flux(
                    flux_odd, err_odd, filtered_stars["fluxratio"].values[i]
                    )
            if has_even_odd:
                # All hypotheses must be compared on the same observations.
                # Ordinary scenarios fit one shared transit shape to both
                # folds; x2P scenarios below receive the two folds separately.
                time, flux, flux_err = _concatenate_even_odd_observations(
                    time_even, flux_even, err_even,
                    time_odd, flux_odd, err_odd,
                )

            M_s = filtered_stars["mass"].values[i]
            R_s = filtered_stars["rad"].values[i]
            Teff = filtered_stars["Teff"].values[i]
            Tmag = filtered_stars["Tmag"].values[i]
            Jmag = filtered_stars["Jmag"].values[i]
            Hmag = filtered_stars["Hmag"].values[i]
            Kmag = filtered_stars["Kmag"].values[i]
            plx = filtered_stars["plx"].values[i]
            Z = 0.0
            ra = filtered_stars["ra"].values[i]
            dec = filtered_stars["dec"].values[i]

            # get url to TRILEGAL results and save
            if self.trilegal_fname is None:
                output_url = self.trilegal_url
                trilegal_fname = save_trilegal(output_url, self.ID)
                # save the downloaded filename for future calc_probs() calls to avoid:
                # 1. repeated download, and
                # 2. HTTP 400 error, if the users use the target instance for a long time
                #   such that TRILEGAL deletes the result.
                self.trilegal_fname = trilegal_fname
            else:
                trilegal_fname = self.trilegal_fname

            # target star
            if i == 0:

                # check to see if there are any missing stellar
                # parameters and, if there are, ask for input
                if (np.isnan(M_s) or np.isnan(R_s)
                    or np.isnan(Teff) or np.isnan(plx)):
                    print(
                        "Insufficient information to validate "
                        + str(ID)
                        + ". Please ensure a stellar "
                        + "mass (in M_Sun), radius (in R_Sun), "
                        + "Teff (in K), and plx (in mas) "
                        + "are provided in the .stars dataframe."
                        )
                    break

                else:
                    if "TP" in drop_scenario:
                        j = 0
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "TP"
                        lnZ[j] = -np.inf
                    else:
                        if verbose == 1:
                            print(
                                "Calculating TP scenario "
                                + "probabilitiey for " + str(ID) + "."
                                )

                        res = lnZ_TTP(
                            time, flux, flux_err, P_orb,
                            M_s, R_s, Teff, Z,
                            *sampling_pair("TP"), self.mission,
                            flatpriors,
                            exptime, nsamples
                            )
                        # self.res_TTP = res
                        j = 0
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "TP"
                        best_M_host[j] = res["M_s"][0]
                        best_R_host[j] = res["R_s"][0]
                        best_u1[j] = res["u1"][0]
                        best_u2[j] = res["u2"][0]
                        best_P_orb[j] = res["P_orb"][0]
                        best_i[j] = res["inc"][0]
                        best_b[j] = res["b"][0]
                        best_R_p[j] = res["R_p"][0]
                        best_ecc[j] = res["ecc"][0]
                        best_argp[j] = res["argp"][0]
                        best_M_EB[j] = res["M_EB"][0]
                        best_R_EB[j] = res["R_EB"][0]
                        best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                        lnZ[j] = res["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                        best_lnL[j] = res.get('best_lnL', res['lnL'][0]- _sec_lnL_correction if 'lnL' in res else np.nan)- _sec_lnL_correction
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            lnZ[j] = lnZ[j] - _sec_lnL_correction

                    if "EB" in drop_scenario:
                        j = 1
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "EB"
                        lnZ[j] = -np.inf
                        j = 2
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "EBx2P"
                        lnZ[j] = -np.inf
                    else:
                        if verbose == 1:
                            print(
                                "Calculating EB and EBx2P scenario "
                                + "probabilities for " + str(ID) + "."
                                )
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            print('Using secondary eclipse data to calculate TEB probabilities.')
                            res = lnZ_TEB_secondary(
                                time, flux, flux_err,
                                 time_secondary, flux_secondary, err_secondary,
                                  P_orb,
                                M_s, R_s, Teff, Z,
                                *sampling_pair("EB"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                secondary_loglike=secondary_loglike,
                                )
                            if flux_even is not None and time_even is not None and err_even is not None and flux_odd is not None and time_odd is not None and err_odd is not None:
                                print('Using odd-even transit data to calculate TEB probabilities.')
                                res_twin = lnZ_TEB_evenodd(
                                      time_even, flux_even, err_even,
                                      time_odd, flux_odd, err_odd,
                                      P_orb,
                                    M_s, R_s, Teff, Z,
                                    *sampling_pair("EBx2P"), self.mission,
                                    flatpriors,
                                    exptime, nsamples
                                    )
                            else:
                                _,res_twin = lnZ_TEB(
                                    time, flux, flux_err, P_orb,
                                    M_s, R_s, Teff, Z,
                                    *sampling_pair("EB"), self.mission,
                                    flatpriors,
                                    exptime, nsamples,
                                    **twin_sampling("EBx2P")
                                    )
                        else:
                            res, res_twin = lnZ_TEB(
                                time, flux, flux_err, P_orb,
                                M_s, R_s, Teff, Z,
                                *sampling_pair("EB"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                **twin_sampling("EBx2P")
                                )
                        # self.res_TEB = res
                        j = 1
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "EB"
                        best_M_host[j] = res["M_s"][0]
                        best_R_host[j] = res["R_s"][0]
                        best_u1[j] = res["u1"][0]
                        best_u2[j] = res["u2"][0]
                        best_P_orb[j] = res["P_orb"][0]
                        best_i[j] = res["inc"][0]
                        best_b[j] = res["b"][0]
                        best_R_p[j] = res["R_p"][0]
                        best_ecc[j] = res["ecc"][0]
                        best_argp[j] = res["argp"][0]
                        best_M_EB[j] = res["M_EB"][0]
                        best_R_EB[j] = res["R_EB"][0]
                        best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                        lnZ[j] = res["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                        best_lnL[j] = res.get('best_lnL', res['lnL'][0] if 'lnL' in res else np.nan)
                        # self.res_TEBx2P = res_twin
                        j = 2
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "EBx2P"
                        best_M_host[j] = res_twin["M_s"][0]
                        best_R_host[j] = res_twin["R_s"][0]
                        best_u1[j] = res_twin["u1"][0]
                        best_u2[j] = res_twin["u2"][0]
                        best_P_orb[j] = res_twin["P_orb"][0]
                        best_i[j] = res_twin["inc"][0]
                        best_b[j] = res_twin["b"][0]
                        best_R_p[j] = res_twin["R_p"][0]
                        best_ecc[j] = res_twin["ecc"][0]
                        best_argp[j] = res_twin["argp"][0]
                        best_M_EB[j] = res_twin["M_EB"][0]
                        best_R_EB[j] = res_twin["R_EB"][0]
                        best_fluxratio_EB[j] = res_twin["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res_twin["fluxratio_comp"][0]
                        lnZ[j] = res_twin["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res_twin)
                        best_lnL[j] = res_twin.get('best_lnL', res_twin['lnL'][0]- _sec_lnL_correction if 'lnL' in res_twin else np.nan)- _sec_lnL_correction
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            lnZ[j] = lnZ[j] - _sec_lnL_correction

                        

                    if "PTP" in drop_scenario:
                        j = 3
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "PTP"
                        lnZ[j] = -np.inf
                    else:
                        if verbose == 1:
                            print(
                                "Calculating PTP scenario "
                                + "probability for " + str(ID) + "."
                                )

                        res = lnZ_PTP(
                            time, flux, flux_err, P_orb,
                            M_s, R_s, Teff, Z,
                            plx, contrast_curve_file,
                            filt,
                            *sampling_pair("PTP"), self.mission,
                            flatpriors,
                            exptime, nsamples,
                            molusc_file
                            )
                        # self.res_PTP = res
                        j = 3
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "PTP"
                        best_M_host[j] = res["M_s"][0]
                        best_R_host[j] = res["R_s"][0]
                        best_u1[j] = res["u1"][0]
                        best_u2[j] = res["u2"][0]
                        best_P_orb[j] = res["P_orb"][0]
                        best_i[j] = res["inc"][0]
                        best_b[j] = res["b"][0]
                        best_R_p[j] = res["R_p"][0]
                        best_ecc[j] = res["ecc"][0]
                        best_argp[j] = res["argp"][0]
                        best_M_EB[j] = res["M_EB"][0]
                        best_R_EB[j] = res["R_EB"][0]
                        best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                        lnZ[j] = res["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                        best_lnL[j] = res.get('best_lnL', res['lnL'][0]- _sec_lnL_correction if 'lnL' in res else np.nan)- _sec_lnL_correction
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            lnZ[j] = lnZ[j] - _sec_lnL_correction


                    if "PEB" in drop_scenario:
                        j = 4
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "PEB"
                        lnZ[j] = -np.inf
                        j = 5
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "PEBx2P"
                        lnZ[j] = -np.inf
                    else:
                        if verbose == 1:
                            print(
                                "Calculating PEB and PEBx2P scenario "
                                + "probabilities for " + str(ID) + "."
                                )
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            print('Using secondary eclipse data to calculate PEB probabilities.')
                            res = lnZ_PEB_secondary(
                                time, flux, flux_err,
                                 time_secondary, flux_secondary, err_secondary,
                                  P_orb,
                                M_s, R_s, Teff, Z,
                                plx, contrast_curve_file,
                                filt,
                                *sampling_pair("PEB"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                molusc_file,
                                secondary_loglike=secondary_loglike,
                                )
                            if flux_even is not None and time_even is not None and err_even is not None and flux_odd is not None and time_odd is not None and err_odd is not None:
                                print('Using odd-even transit data to calculate PEB probabilities.')
                                res_twin = lnZ_PEB_evenodd(
                                      time_even, flux_even, err_even,
                                      time_odd, flux_odd, err_odd,
                                      P_orb,
                                      M_s, R_s, Teff, Z,
                                plx, contrast_curve_file,
                                filt,
                                *sampling_pair("PEBx2P"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                molusc_file
                                    )
                            else:
                                _,res_twin = lnZ_PEB(
                                    time, flux, flux_err, P_orb,
                                    M_s, R_s, Teff, Z,
                                    plx, contrast_curve_file,
                                    filt,
                                    *sampling_pair("PEB"), self.mission,
                                    flatpriors,
                                    exptime, nsamples,
                                    molusc_file,
                                    **twin_sampling("PEBx2P")
                                    )
                        else:
                            res, res_twin = lnZ_PEB(
                                time, flux, flux_err, P_orb,
                                M_s, R_s, Teff, Z,
                                plx, contrast_curve_file,
                                filt,
                                *sampling_pair("PEB"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                molusc_file,
                                **twin_sampling("PEBx2P")
                                )
                        # self.res_PEB = res
                        j = 4
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "PEB"
                        best_M_host[j] = res["M_s"][0]
                        best_R_host[j] = res["R_s"][0]
                        best_u1[j] = res["u1"][0]
                        best_u2[j] = res["u2"][0]
                        best_P_orb[j] = res["P_orb"][0]
                        best_i[j] = res["inc"][0]
                        best_b[j] = res["b"][0]
                        best_R_p[j] = res["R_p"][0]
                        best_ecc[j] = res["ecc"][0]
                        best_argp[j] = res["argp"][0]
                        best_M_EB[j] = res["M_EB"][0]
                        best_R_EB[j] = res["R_EB"][0]
                        best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                        lnZ[j] = res["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                        best_lnL[j] = res.get('best_lnL', res['lnL'][0] if 'lnL' in res else np.nan)
                        # self.res_PEBx2P = res_twin
                        j = 5
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "PEBx2P"
                        best_M_host[j] = res_twin["M_s"][0]
                        best_R_host[j] = res_twin["R_s"][0]
                        best_u1[j] = res_twin["u1"][0]
                        best_u2[j] = res_twin["u2"][0]
                        best_P_orb[j] = res_twin["P_orb"][0]
                        best_i[j] = res_twin["inc"][0]
                        best_b[j] = res_twin["b"][0]
                        best_R_p[j] = res_twin["R_p"][0]
                        best_ecc[j] = res_twin["ecc"][0]
                        best_argp[j] = res_twin["argp"][0]
                        best_M_EB[j] = res_twin["M_EB"][0]
                        best_R_EB[j] = res_twin["R_EB"][0]
                        best_fluxratio_EB[j] = res_twin["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res_twin["fluxratio_comp"][0]
                        lnZ[j] = res_twin["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res_twin)
                        best_lnL[j] = res_twin.get('best_lnL', res_twin['lnL'][0]- _sec_lnL_correction if 'lnL' in res_twin else np.nan)- _sec_lnL_correction
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            lnZ[j] = lnZ[j] - _sec_lnL_correction


                    if "STP" in drop_scenario:
                        j = 6
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "STP"
                        lnZ[j] = -np.inf
                    else:
                        if verbose == 1:
                            print(
                                "Calculating STP scenario "
                                + "probability for " + str(ID) + "."
                                )

                        res = lnZ_STP(
                            time, flux, flux_err, P_orb,
                            M_s, R_s, Teff, Z,
                            plx, contrast_curve_file,
                            filt,
                            *sampling_pair("STP"), self.mission,
                            flatpriors,
                            exptime, nsamples,
                            molusc_file
                            )
                        # self.res_STP = res
                        j = 6
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "STP"
                        best_M_host[j] = res["M_s"][0]
                        best_R_host[j] = res["R_s"][0]
                        best_u1[j] = res["u1"][0]
                        best_u2[j] = res["u2"][0]
                        best_P_orb[j] = res["P_orb"][0]
                        best_i[j] = res["inc"][0]
                        best_b[j] = res["b"][0]
                        best_R_p[j] = res["R_p"][0]
                        best_ecc[j] = res["ecc"][0]
                        best_argp[j] = res["argp"][0]
                        best_M_EB[j] = res["M_EB"][0]
                        best_R_EB[j] = res["R_EB"][0]
                        best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                        lnZ[j] = res["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                        best_lnL[j] = res.get('best_lnL', res['lnL'][0]- _sec_lnL_correction if 'lnL' in res else np.nan)- _sec_lnL_correction
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            lnZ[j] = lnZ[j] - _sec_lnL_correction


                    if "SEB" in drop_scenario:
                        j = 7
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "SEB"
                        lnZ[j] = -np.inf
                        j = 8
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "SEBx2P"
                        lnZ[j] = -np.inf
                    else:
                        if verbose == 1:
                            print(
                                "Calculating SEB and SEBx2P scenario "
                                + "probabilities for " + str(ID) + "."
                                )
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            print('Using secondary eclipse data to calculate SEB probabilities.')
                            res = lnZ_SEB_secondary(
                                time, flux, flux_err,
                                 time_secondary, flux_secondary, err_secondary,
                                  P_orb,
                                M_s, R_s, Teff, Z,
                                plx, contrast_curve_file,
                                filt,
                                *sampling_pair("SEB"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                molusc_file,
                                secondary_loglike=secondary_loglike,
                                )
                            if flux_even is not None and time_even is not None and err_even is not None and flux_odd is not None and time_odd is not None and err_odd is not None:
                                print('Using odd-even transit data to calculate SEB probabilities.')
                                res_twin = lnZ_SEB_evenodd(
                                      time_even, flux_even, err_even,
                                      time_odd, flux_odd, err_odd,
                                      P_orb,
                                      M_s, R_s, Teff, Z,
                                plx, contrast_curve_file,
                                filt,
                                *sampling_pair("SEBx2P"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                molusc_file
                                    )
                            else:
                                _,res_twin = lnZ_SEB(
                                time, flux, flux_err, P_orb,
                                M_s, R_s, Teff, Z,
                                plx, contrast_curve_file,
                                filt,
                                *sampling_pair("SEB"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                molusc_file,
                                **twin_sampling("SEBx2P")
                                )
                        else:   
                            res, res_twin = lnZ_SEB(
                                time, flux, flux_err, P_orb,
                                M_s, R_s, Teff, Z,
                                plx, contrast_curve_file,
                                filt,
                                *sampling_pair("SEB"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                molusc_file,
                                **twin_sampling("SEBx2P")
                                )
                        # self.res_SEB = res
                        j = 7
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "SEB"
                        best_M_host[j] = res["M_s"][0]
                        best_R_host[j] = res["R_s"][0]
                        best_u1[j] = res["u1"][0]
                        best_u2[j] = res["u2"][0]
                        best_P_orb[j] = res["P_orb"][0]
                        best_i[j] = res["inc"][0]
                        best_b[j] = res["b"][0]
                        best_R_p[j] = res["R_p"][0]
                        best_ecc[j] = res["ecc"][0]
                        best_argp[j] = res["argp"][0]
                        best_M_EB[j] = res["M_EB"][0]
                        best_R_EB[j] = res["R_EB"][0]
                        best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                        lnZ[j] = res["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                        best_lnL[j] = res.get('best_lnL', res['lnL'][0] if 'lnL' in res else np.nan)
                        # self.res_SEBx2P = res_twin
                        j = 8
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "SEBx2P"
                        best_M_host[j] = res_twin["M_s"][0]
                        best_R_host[j] = res_twin["R_s"][0]
                        best_u1[j] = res_twin["u1"][0]
                        best_u2[j] = res_twin["u2"][0]
                        best_P_orb[j] = res_twin["P_orb"][0]
                        best_i[j] = res_twin["inc"][0]
                        best_b[j] = res_twin["b"][0]
                        best_R_p[j] = res_twin["R_p"][0]
                        best_ecc[j] = res_twin["ecc"][0]
                        best_argp[j] = res_twin["argp"][0]
                        best_M_EB[j] = res_twin["M_EB"][0]
                        best_R_EB[j] = res_twin["R_EB"][0]
                        best_fluxratio_EB[j] = res_twin["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res_twin["fluxratio_comp"][0]
                        lnZ[j] = res_twin["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res_twin)
                        best_lnL[j] = res_twin.get('best_lnL', res_twin['lnL'][0]- _sec_lnL_correction if 'lnL' in res_twin else np.nan)- _sec_lnL_correction
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            lnZ[j] = lnZ[j] - _sec_lnL_correction


                    if "DTP" in drop_scenario:
                        j = 9
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "DTP"
                        lnZ[j] = -np.inf
                    else:
                        if verbose == 1:
                            print(
                                "Calculating DTP scenario "
                                + "probability for " + str(ID) + "."
                                )

                        res = lnZ_DTP(
                            time, flux, flux_err, P_orb,
                            M_s, R_s, Teff, Z,
                            Tmag, Jmag, Hmag, Kmag,
                            trilegal_fname,
                            contrast_curve_file, filt,
                            *sampling_pair("DTP"), self.mission,
                            flatpriors,
                            exptime, nsamples
                            )
                        # self.res_DTP = res
                        j = 9
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "DTP"
                        best_M_host[j] = res["M_s"][0]
                        best_R_host[j] = res["R_s"][0]
                        best_u1[j] = res["u1"][0]
                        best_u2[j] = res["u2"][0]
                        best_P_orb[j] = res["P_orb"][0]
                        best_i[j] = res["inc"][0]
                        best_b[j] = res["b"][0]
                        best_R_p[j] = res["R_p"][0]
                        best_ecc[j] = res["ecc"][0]
                        best_argp[j] = res["argp"][0]
                        best_M_EB[j] = res["M_EB"][0]
                        best_R_EB[j] = res["R_EB"][0]
                        best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                        lnZ[j] = res["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                        best_lnL[j] = res.get('best_lnL', res['lnL'][0]- _sec_lnL_correction if 'lnL' in res else np.nan)- _sec_lnL_correction
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            lnZ[j] = lnZ[j] - _sec_lnL_correction


                    if "DEB" in drop_scenario:
                        j = 10
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "DEB"
                        lnZ[j] = -np.inf
                        j = 11
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "DEBx2P"
                        lnZ[j] = -np.inf
                    else:
                        if verbose == 1:
                            print(
                                "Calculating DEB and DEBx2P scenario "
                                + "probabilities for " + str(ID) + "."
                                )
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            print('Using secondary eclipse data to calculate DEB probabilities.')
                            res = lnZ_DEB_secondary(
                                time, flux, flux_err,
                                 time_secondary, flux_secondary, err_secondary,
                                  P_orb,
                                M_s, R_s, Teff, Z,
                                Tmag, Jmag, Hmag, Kmag,
                                trilegal_fname,
                                contrast_curve_file, filt,
                                *sampling_pair("DEB"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                secondary_loglike=secondary_loglike,
                                )
                            if flux_even is not None and time_even is not None and err_even is not None and flux_odd is not None and time_odd is not None and err_odd is not None:
                                print('Using odd-even transit data to calculate DEB probabilities.')
                                res_twin = lnZ_DEB_evenodd(
                                      time_even, flux_even, err_even,
                                      time_odd, flux_odd, err_odd,
                                      P_orb,
                                      M_s, R_s, Teff, Z,
                                Tmag, Jmag, Hmag, Kmag,
                                trilegal_fname,
                                contrast_curve_file, filt,
                                *sampling_pair("DEBx2P"), self.mission,
                                flatpriors,
                                exptime, nsamples
                                    )
                            else:
                                _,res_twin = lnZ_DEB(
                                    time, flux, flux_err, P_orb,
                                    M_s, R_s, Teff, Z,
                                    Tmag, Jmag, Hmag, Kmag,
                                    trilegal_fname,
                                    contrast_curve_file, filt,
                                    *sampling_pair("DEB"), self.mission,
                                    flatpriors,
                                    exptime, nsamples,
                                    **twin_sampling("DEBx2P")
                                    )
                        else:
                            res, res_twin = lnZ_DEB(
                                time, flux, flux_err, P_orb,
                                M_s, R_s, Teff, Z,
                                Tmag, Jmag, Hmag, Kmag,
                                trilegal_fname,
                                contrast_curve_file, filt,
                                *sampling_pair("DEB"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                **twin_sampling("DEBx2P")
                                )
                        # self.res_DEB = res
                        j = 10
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "DEB"
                        best_M_host[j] = res["M_s"][0]
                        best_R_host[j] = res["R_s"][0]
                        best_u1[j] = res["u1"][0]
                        best_u2[j] = res["u2"][0]
                        best_P_orb[j] = res["P_orb"][0]
                        best_i[j] = res["inc"][0]
                        best_b[j] = res["b"][0]
                        best_R_p[j] = res["R_p"][0]
                        best_ecc[j] = res["ecc"][0]
                        best_argp[j] = res["argp"][0]
                        best_M_EB[j] = res["M_EB"][0]
                        best_R_EB[j] = res["R_EB"][0]
                        best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                        lnZ[j] = res["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                        best_lnL[j] = res.get('best_lnL', res['lnL'][0] if 'lnL' in res else np.nan)- _sec_lnL_correction
                        # self.res_DEBx2P = res_twin
                        j = 11
                        targets[j] = ID
                        star_num[j] = 1
                        scenarios[j] = "DEBx2P"
                        best_M_host[j] = res_twin["M_s"][0]
                        best_R_host[j] = res_twin["R_s"][0]
                        best_u1[j] = res_twin["u1"][0]
                        best_u2[j] = res_twin["u2"][0]
                        best_P_orb[j] = res_twin["P_orb"][0]
                        best_i[j] = res_twin["inc"][0]
                        best_b[j] = res_twin["b"][0]
                        best_R_p[j] = res_twin["R_p"][0]
                        best_ecc[j] = res_twin["ecc"][0]
                        best_argp[j] = res_twin["argp"][0]
                        best_M_EB[j] = res_twin["M_EB"][0]
                        best_R_EB[j] = res_twin["R_EB"][0]
                        best_fluxratio_EB[j] = res_twin["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res_twin["fluxratio_comp"][0]
                        lnZ[j] = res_twin["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res_twin)
                        best_lnL[j] = res_twin.get('best_lnL', res_twin['lnL'][0]- _sec_lnL_correction if 'lnL' in res_twin else np.nan)- _sec_lnL_correction
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            lnZ[j] = lnZ[j] - _sec_lnL_correction


                    if "BTP" in drop_scenario:
                        j = 12
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "BTP"
                        lnZ[j] = -np.inf
                    else:
                        if verbose == 1:
                            print(
                                "Calculating BTP scenario "
                                + "probability for " + str(ID) + "."
                                )

                        res = lnZ_BTP(
                            time, flux, flux_err, P_orb,
                            M_s, R_s, Teff,
                            Tmag, Jmag, Hmag, Kmag,
                            trilegal_fname,
                            contrast_curve_file, filt,
                            *sampling_pair("BTP"), self.mission,
                            flatpriors,
                            exptime, nsamples,
                            )
                        # self.res_BTP = res
                        j = 12
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "BTP"
                        best_M_host[j] = res["M_s"][0]
                        best_R_host[j] = res["R_s"][0]
                        best_u1[j] = res["u1"][0]
                        best_u2[j] = res["u2"][0]
                        best_P_orb[j] = res["P_orb"][0]
                        best_i[j] = res["inc"][0]
                        best_b[j] = res["b"][0]
                        best_R_p[j] = res["R_p"][0]
                        best_ecc[j] = res["ecc"][0]
                        best_argp[j] = res["argp"][0]
                        best_M_EB[j] = res["M_EB"][0]
                        best_R_EB[j] = res["R_EB"][0]
                        best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                        lnZ[j] = res["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                        best_lnL[j] = res.get('best_lnL', res['lnL'][0]- _sec_lnL_correction if 'lnL' in res else np.nan)- _sec_lnL_correction
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            lnZ[j] = lnZ[j] - _sec_lnL_correction


                    if "BEB" in drop_scenario:
                        j = 13
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "BEB"
                        lnZ[j] = -np.inf
                        j = 14
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "BEBx2P"
                        lnZ[j] = -np.inf
                    else:
                        if verbose == 1:
                            print(
                                "Calculating BEB and BEBx2P scenario "
                                + "probabilities for " + str(ID) + "."
                                )
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            print('Using secondary eclipse data to calculate BEB probabilities.')
                            res = lnZ_BEB_secondary(
                                time, flux, flux_err,
                                 time_secondary, flux_secondary, err_secondary,
                                  P_orb,
                                M_s, R_s, Teff,
                                Tmag, Jmag, Hmag, Kmag,
                                trilegal_fname,
                                contrast_curve_file, filt,
                                *sampling_pair("BEB"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                secondary_loglike=secondary_loglike,
                                )
                            if flux_even is not None and time_even is not None and err_even is not None and flux_odd is not None and time_odd is not None and err_odd is not None:
                                print('Using odd-even transit data to calculate BEB probabilities.')
                                res_twin = lnZ_BEB_evenodd(
                                      time_even, flux_even, err_even,
                                      time_odd, flux_odd, err_odd,
                                      P_orb,
                                      M_s, R_s, Teff,
                                Tmag, Jmag, Hmag, Kmag,
                                trilegal_fname,
                                contrast_curve_file, filt,
                                *sampling_pair("BEBx2P"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                    )
                            else:
                                _,res_twin = lnZ_BEB(
                                    time, flux, flux_err, P_orb,
                                    M_s, R_s, Teff,
                                    Tmag, Jmag, Hmag, Kmag,
                                    trilegal_fname,
                                    contrast_curve_file, filt,
                                    *sampling_pair("BEB"), self.mission,
                                    flatpriors,
                                    exptime, nsamples,
                                    **twin_sampling("BEBx2P")
                                    )
                        else:
                            res, res_twin = lnZ_BEB(
                                time, flux, flux_err, P_orb,
                                M_s, R_s, Teff,
                                Tmag, Jmag, Hmag, Kmag,
                                trilegal_fname,
                                contrast_curve_file, filt,
                                *sampling_pair("BEB"), self.mission,
                                flatpriors,
                                exptime, nsamples,
                                **twin_sampling("BEBx2P")
                                )
                        # self.res_BEB = res
                        j = 13
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "BEB"
                        best_M_host[j] = res["M_s"][0]
                        best_R_host[j] = res["R_s"][0]
                        best_u1[j] = res["u1"][0]
                        best_u2[j] = res["u2"][0]
                        best_P_orb[j] = res["P_orb"][0]
                        best_i[j] = res["inc"][0]
                        best_b[j] = res["b"][0]
                        best_R_p[j] = res["R_p"][0]
                        best_ecc[j] = res["ecc"][0]
                        best_argp[j] = res["argp"][0]
                        best_M_EB[j] = res["M_EB"][0]
                        best_R_EB[j] = res["R_EB"][0]
                        best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                        lnZ[j] = res["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                        best_lnL[j] = res.get('best_lnL', res['lnL'][0] if 'lnL' in res else np.nan)
                        # self.res_BEBx2P = res_twin
                        j = 14
                        targets[j] = ID
                        star_num[j] = 2
                        scenarios[j] = "BEBx2P"
                        best_M_host[j] = res_twin["M_s"][0]
                        best_R_host[j] = res_twin["R_s"][0]
                        best_u1[j] = res_twin["u1"][0]
                        best_u2[j] = res_twin["u2"][0]
                        best_P_orb[j] = res_twin["P_orb"][0]
                        best_i[j] = res_twin["inc"][0]
                        best_b[j] = res_twin["b"][0]
                        best_R_p[j] = res_twin["R_p"][0]
                        best_ecc[j] = res_twin["ecc"][0]
                        best_argp[j] = res_twin["argp"][0]
                        best_M_EB[j] = res_twin["M_EB"][0]
                        best_R_EB[j] = res_twin["R_EB"][0]
                        best_fluxratio_EB[j] = res_twin["fluxratio_EB"][0]
                        best_fluxratio_comp[j] = res_twin["fluxratio_comp"][0]
                        lnZ[j] = res_twin["lnZ"]
                        if output_posteriors: self._stash_posterior(j, scenarios, targets, res_twin)
                        best_lnL[j] = res_twin.get('best_lnL', res_twin['lnL'][0]- _sec_lnL_correction if 'lnL' in res_twin else np.nan)- _sec_lnL_correction
                        if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                            lnZ[j] = lnZ[j] - _sec_lnL_correction


            # nearby stars
            else:
                # if a property is unknown, assume solar value
                if np.isnan(Teff):
                    Teff = 5777
                if np.isnan(M_s):
                    M_s = 1.0
                if np.isnan(R_s):
                    R_s = 1.0
                if verbose == 1:
                    print(
                        "Calculating NTP, NEB, and NEB2xP scenario "
                        + "probabilities for " + str(ID) + "."
                        )

                res = lnZ_TTP(
                    time, flux, flux_err, P_orb,
                    M_s, R_s, Teff, Z,
                    *sampling_pair("NTP"), self.mission,
                    flatpriors,
                    exptime, nsamples,
                    )
                j = 15 + 3*(i-1)
                targets[j] = ID
                star_num[j] = 1
                scenarios[j] = "NTP"
                best_M_host[j] = res["M_s"][0]
                best_R_host[j] = res["R_s"][0]
                best_u1[j] = res["u1"][0]
                best_u2[j] = res["u2"][0]
                best_P_orb[j] = res["P_orb"][0]
                best_i[j] = res["inc"][0]
                best_b[j] = res["b"][0]
                best_R_p[j] = res["R_p"][0]
                best_ecc[j] = res["ecc"][0]
                best_argp[j] = res["argp"][0]
                best_M_EB[j] = res["M_EB"][0]
                best_R_EB[j] = res["R_EB"][0]
                best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                lnZ[j] = res["lnZ"]
                if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                best_lnL[j] = res.get('best_lnL', res['lnL'][0]- _sec_lnL_correction if 'lnL' in res else np.nan)- _sec_lnL_correction
                if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                    lnZ[j] = lnZ[j] - _sec_lnL_correction


                if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                    res = lnZ_TEB_secondary(
                        time, flux, flux_err,
                        time_secondary, flux_secondary, err_secondary, P_orb,
                        M_s, R_s, Teff, Z,
                        *sampling_pair("NEB"), self.mission,
                        flatpriors,
                        exptime, nsamples,
                        secondary_loglike=secondary_loglike,
                        )
                    if flux_even is not None and time_even is not None and err_even is not None and flux_odd is not None and time_odd is not None and err_odd is not None:
                        print('Using odd-even transit data to calculate TEB probabilities.')
                        res_twin = lnZ_TEB_evenodd(
                              time_even, flux_even, err_even,
                              time_odd, flux_odd, err_odd,
                              P_orb,
                              M_s, R_s, Teff, Z,
                        *sampling_pair("NEBx2P"), self.mission,
                        flatpriors,
                        exptime, nsamples,
                            )
                    else:
                        _,res_twin = lnZ_TEB(
                            time, flux, flux_err, P_orb,
                            M_s, R_s, Teff, Z,
                            *sampling_pair("NEB"), self.mission,
                            flatpriors,
                            exptime, nsamples,
                            **twin_sampling("NEBx2P")
                            )
                else:
                    res, res_twin = lnZ_TEB(
                            time, flux, flux_err, P_orb,
                            M_s, R_s, Teff, Z,
                            *sampling_pair("NEB"), self.mission,
                            flatpriors,
                            exptime, nsamples,
                            **twin_sampling("NEBx2P")
                            )

                j = 16 + 3*(i-1)
                targets[j] = ID
                star_num[j] = 1
                scenarios[j] = "NEB"
                best_M_host[j] = res["M_s"][0]
                best_R_host[j] = res["R_s"][0]
                best_u1[j] = res["u1"][0]
                best_u2[j] = res["u2"][0]
                best_P_orb[j] = res["P_orb"][0]
                best_i[j] = res["inc"][0]
                best_b[j] = res["b"][0]
                best_R_p[j] = res["R_p"][0]
                best_ecc[j] = res["ecc"][0]
                best_argp[j] = res["argp"][0]
                best_M_EB[j] = res["M_EB"][0]
                best_R_EB[j] = res["R_EB"][0]
                best_fluxratio_EB[j] = res["fluxratio_EB"][0]
                best_fluxratio_comp[j] = res["fluxratio_comp"][0]
                lnZ[j] = res["lnZ"]
                if output_posteriors: self._stash_posterior(j, scenarios, targets, res)
                best_lnL[j] = res.get('best_lnL', res['lnL'][0] if 'lnL' in res else np.nan)
                j = 17 + 3*(i-1)
                targets[j] = ID
                star_num[j] = 1
                scenarios[j] = "NEBx2P"
                best_M_host[j] = res_twin["M_s"][0]
                best_R_host[j] = res_twin["R_s"][0]
                best_u1[j] = res_twin["u1"][0]
                best_u2[j] = res_twin["u2"][0]
                best_P_orb[j] = res_twin["P_orb"][0]
                best_i[j] = res_twin["inc"][0]
                best_b[j] = res_twin["b"][0]
                best_R_p[j] = res_twin["R_p"][0]
                best_ecc[j] = res_twin["ecc"][0]
                best_argp[j] = res_twin["argp"][0]
                best_M_EB[j] = res_twin["M_EB"][0]
                best_R_EB[j] = res_twin["R_EB"][0]
                best_fluxratio_EB[j] = res_twin["fluxratio_EB"][0]
                best_fluxratio_comp[j] = res_twin["fluxratio_comp"][0]
                lnZ[j] = res_twin["lnZ"]
                if output_posteriors: self._stash_posterior(j, scenarios, targets, res_twin)
                best_lnL[j] = res_twin.get('best_lnL', res_twin['lnL'][0]- _sec_lnL_correction if 'lnL' in res_twin else np.nan)- _sec_lnL_correction
                if flux_secondary is not None and time_secondary is not None and err_secondary is not None:
                    lnZ[j] = lnZ[j] - _sec_lnL_correction


        # calculate the relative probability of each scenario
        # use log-sum-exp trick to avoid underflow
        lnZ_max = np.max(lnZ[lnZ > -np.inf]) if np.any(lnZ > -np.inf) else 0.0
        relative_probs = np.exp(lnZ - lnZ_max)
        relative_probs /= np.sum(relative_probs)

        # now save all of the arrays as a dataframe
        prob_df = DataFrame({
            "ID": targets,
            "scenario": scenarios,
            "M_s": best_M_host,
            "R_s": best_R_host,
            "P_orb": best_P_orb,
            "inc": best_i,
            "b": best_b,
            "ecc": best_ecc,
            "w": best_argp,
            "R_p": best_R_p,
            "M_EB": best_M_EB,
            "R_EB": best_R_EB,
            "prob": relative_probs,
            "lnZ": lnZ,
            "best_lnL": best_lnL,
            "sampling_N": [self.sampling_config[s]["N"] if s in self.sampling_config else np.nan for s in scenarios],
            "sampling_steps": [self.sampling_config[s]["steps"] if s in self.sampling_config else np.nan for s in scenarios],
            "sampling_policy": POLICY_VERSION,
            })
        self.probs = prob_df
        self.star_num = star_num
        self.u1 = best_u1
        self.u2 = best_u2
        self.fluxratio_EB = best_fluxratio_EB
        self.fluxratio_comp = best_fluxratio_comp

        # calculate the FPP and EBP
        self.FPP = 1-(prob_df.prob[0]+prob_df.prob[3]+prob_df.prob[9])
        if len(prob_df.prob) > 15:
            self.NFPP = np.sum(prob_df.prob[15:])
        else:
            self.NFPP = 0.0

        return

    def plot_fits(self, time: np.ndarray,
                  flux_0: np.ndarray, flux_err_0: float,
                  save: bool = False, fname: str = None,
                  time_secondary: np.ndarray = None,
                  flux_secondary: np.ndarray = None,
                  err_secondary: float = None,
                  time_even: np.ndarray = None,
                  flux_even: np.ndarray = None,
                  err_even: float = None,
                  time_odd: np.ndarray = None,
                  flux_odd: np.ndarray = None,
                  err_odd: float = None):
        """Visualize best-fit for each scenarios.

        Plots light curve for best-fit instance of each scenario.
        Layout: 5 columns per row —
          col 0: TP (primary transit)
          col 1: EB primary eclipse
          col 2: EB secondary eclipse
          col 3: EBx2P even transit (or primary if no even/odd data)
          col 4: EBx2P odd transit (or secondary if no even/odd data)

        Args:
            time (numpy array): Time of each data point
                [days from transit midpoint].
            flux_0 (numpy array): Normalized flux of each data point.
            flux_err_0 (float): Uncertainty of flux.
            save (bool): Whether or not to save plot as pdf.
            fname (str): File name of pdf.
            time_secondary (numpy array): Time for secondary eclipse
                data [days from secondary midpoint].
            flux_secondary (numpy array): Normalized flux of secondary
                eclipse data.
            err_secondary (float): Uncertainty of secondary flux.
            time_even (numpy array): Time for even-transit data.
            flux_even (numpy array): Normalized flux of even transits.
            err_even (float): Uncertainty of even-transit flux.
            time_odd (numpy array): Time for odd-transit data.
            flux_odd (numpy array): Normalized flux of odd transits.
            err_odd (float): Uncertainty of odd-transit flux.
        """
        scenario_idx = self.probs[self.probs["ID"] != 0].index.values
        df = self.probs[self.probs["ID"] != 0]
        star_num = self.star_num[self.probs["ID"] != 0]
        u1s = self.u1[self.probs["ID"] != 0]
        u2s = self.u2[self.probs["ID"] != 0]
        fluxratios_EB = self.fluxratio_EB[self.probs["ID"] != 0]
        fluxratios_comp = self.fluxratio_comp[self.probs["ID"] != 0]

        model_time = np.linspace(min(time), max(time), 100)

        # secondary eclipse model time grid
        has_sec_data = (time_secondary is not None
                        and flux_secondary is not None
                        and err_secondary is not None)
        if has_sec_data:
            sec_model_time = np.linspace(
                min(time_secondary), max(time_secondary), 100)

        # even/odd transit data for x2P scenarios
        has_evenodd = (time_even is not None and flux_even is not None
                       and err_even is not None
                       and time_odd is not None and flux_odd is not None
                       and err_odd is not None)
        if has_evenodd:
            even_model_time = np.linspace(min(time_even), max(time_even), 100)
            odd_model_time = np.linspace(min(time_odd), max(time_odd), 100)

        n_rows = len(df) // 3
        n_cols = 5
        f, ax = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 4, n_rows * 3.5)
            )
        if n_rows == 1:
            ax = ax[np.newaxis, :]

        # Colour scheme: neutral grey data, navy primary/even model, amber
        # secondary/odd model.
        DATA_C, PRI_C, SEC_C = "0.6", "#1f4e79", "#d98c2b"

        for i in range(n_rows):
            # index into the original 3-per-row df layout
            k_tp = 3 * i + 0
            k_eb = 3 * i + 1
            k_twin = 3 * i + 2

            for col, k, is_twin in [(0, k_tp, False),
                                     (1, k_eb, False),
                                     (3, k_twin, True)]:
                # subtract flux from other stars in the aperture
                star_idx = np.argwhere(
                    self.stars["ID"].values == str(df["ID"].values[k])
                    )[0, 0]
                flux, flux_err_val = renorm_flux(
                    flux_0, flux_err_0,
                    self.stars["fluxratio"].values[star_idx]
                    )

                if col == 0:
                    # TP scenario
                    if star_num[k] == 1:
                        comp = False
                    else:
                        comp = True
                    a = (
                        (G * df["M_s"].values[k] * Msun) / (4 * pi**2)
                        * (df['P_orb'].values[k] * 86400)**2
                        )**(1/3)
                    u1, u2 = u1s[k], u2s[k]
                    if df["M_s"].values[k] != 0.0 and df["R_s"].values[k] > 0.0:
                        best_model = simulate_TP_transit(
                            model_time,
                            df['R_p'].values[k], df['P_orb'].values[k],
                            df['inc'].values[k], a, df["R_s"].values[k],
                            u1, u2,
                            df["ecc"].values[k], df["w"].values[k],
                            fluxratios_comp[k], comp
                            )
                    else:
                        best_model = np.ones(len(model_time))

                    y_fmt = ticker.ScalarFormatter(useOffset=False)
                    ax[i, col].yaxis.set_major_formatter(y_fmt)
                    ax[i, col].errorbar(
                        time, flux, flux_err_val, fmt=".", ms=3,
                        color=DATA_C, alpha=0.5, elinewidth=0.7, zorder=0,
                        rasterized=True)
                    ax[i, col].plot(
                        model_time, best_model, "-", color=PRI_C, lw=1.8,
                        zorder=2)
                    ax[i, col].set_ylabel("normalized flux", fontsize=10)
                    ax[i, col].annotate(
                        str(df["scenario"].values[k]), xy=(0.05, 0.06),
                        xycoords="axes fraction", fontsize=10)

                else:
                    # EB scenario (col 1/3 = primary, col 2/4 = secondary)
                    sec_col = col + 1
                    if star_num[k] == 1:
                        comp = False
                    else:
                        comp = True
                    mass = df["M_s"].values[k] + df["M_EB"].values[k]
                    P_orb_k = df['P_orb'].values[k]
                    ecc_k = df["ecc"].values[k]
                    argp_k = df["w"].values[k]
                    a = (
                        (G * mass * Msun) / (4 * pi**2)
                        * (P_orb_k * 86400)**2
                        )**(1/3)
                    u1, u2 = u1s[k], u2s[k]

                    # For x2P scenarios with even/odd data available,
                    # plot even and odd transit windows instead of
                    # primary/secondary.
                    use_evenodd = is_twin and has_evenodd

                    if use_evenodd:
                        # --- even/odd for x2P ---
                        flux_even_renorm, err_even_renorm = renorm_flux(
                            flux_even, err_even,
                            self.stars["fluxratio"].values[star_idx]
                            )
                        flux_odd_renorm, err_odd_renorm = renorm_flux(
                            flux_odd, err_odd,
                            self.stars["fluxratio"].values[star_idx]
                            )
                        if df["M_s"].values[k] != 0.0 and df["R_s"].values[k] > 0.0:
                            even_model, odd_model = \
                                simulate_EB_transit_evenodd(
                                    even_model_time, odd_model_time,
                                    df["R_EB"].values[k], fluxratios_EB[k],
                                    P_orb_k, df['inc'].values[k],
                                    a, df["R_s"].values[k], u1, u2,
                                    ecc_k, argp_k,
                                    fluxratios_comp[k], comp
                                    )
                            # Try both assignments and pick the better fit
                            chi2_fwd = (np.sum((flux_even_renorm
                                        - np.interp(time_even, even_model_time,
                                                    even_model))**2)
                                        + np.sum((flux_odd_renorm
                                        - np.interp(time_odd, odd_model_time,
                                                    odd_model))**2))
                            chi2_rev = (np.sum((flux_even_renorm
                                        - np.interp(time_even, even_model_time,
                                                    odd_model))**2)
                                        + np.sum((flux_odd_renorm
                                        - np.interp(time_odd, odd_model_time,
                                                    even_model))**2))
                            if chi2_rev < chi2_fwd:
                                even_model, odd_model = odd_model, even_model
                        else:
                            even_model = np.ones(len(even_model_time))
                            odd_model = np.ones(len(odd_model_time))

                        # even panel
                        y_fmt = ticker.ScalarFormatter(useOffset=False)
                        ax[i, col].yaxis.set_major_formatter(y_fmt)
                        ax[i, col].errorbar(
                            time_even, flux_even_renorm, err_even_renorm,
                            fmt=".", ms=3, color=DATA_C, alpha=0.5,
                            elinewidth=0.7, zorder=0, rasterized=True)
                        ax[i, col].plot(
                            even_model_time, even_model, "-", color=PRI_C,
                            lw=1.8, zorder=2)
                        ax[i, col].annotate(
                            str(df["scenario"].values[k]) + " even",
                            xy=(0.05, 0.06),
                            xycoords="axes fraction", fontsize=10)

                        # odd panel
                        y_fmt2 = ticker.ScalarFormatter(useOffset=False)
                        ax[i, sec_col].yaxis.set_major_formatter(y_fmt2)
                        ax[i, sec_col].errorbar(
                            time_odd, flux_odd_renorm, err_odd_renorm,
                            fmt=".", ms=3, color=DATA_C, alpha=0.5,
                            elinewidth=0.7, zorder=0, rasterized=True)
                        ax[i, sec_col].plot(
                            odd_model_time, odd_model, "-", color=SEC_C,
                            lw=1.8, zorder=2)
                        ax[i, sec_col].annotate(
                            str(df["scenario"].values[k]) + " odd",
                            xy=(0.05, 0.06),
                            xycoords="axes fraction", fontsize=10)

                    else:
                        # --- primary/secondary for EB (non-twin or no
                        #     even/odd data) ---
                        if df["M_s"].values[k] != 0.0 and df["R_s"].values[k] > 0.0:
                            best_model = simulate_EB_transit(
                                model_time,
                                df["R_EB"].values[k], fluxratios_EB[k],
                                P_orb_k, df['inc'].values[k],
                                a, df["R_s"].values[k], u1, u2,
                                ecc_k, argp_k,
                                fluxratios_comp[k], comp
                                )[0]
                            if has_sec_data:
                                _, sec_model = simulate_EB_transit_secondary(
                                    model_time, sec_model_time,
                                    df["R_EB"].values[k], fluxratios_EB[k],
                                    P_orb_k, df['inc'].values[k],
                                    a, df["R_s"].values[k], u1, u2,
                                    ecc_k, argp_k,
                                    fluxratios_comp[k], comp
                                    )
                            else:
                                sec_time_offset = (
                                    mean_anomaly_difference(
                                        ecc_k, argp_k * (pi / 180.))
                                    - 0.5) * P_orb_k
                                auto_sec_time = np.linspace(
                                    sec_time_offset - abs(max(time)-min(time)),
                                    sec_time_offset + abs(max(time)-min(time)),
                                    100)
                                _, sec_model = simulate_EB_transit_secondary(
                                    model_time, auto_sec_time,
                                    df["R_EB"].values[k], fluxratios_EB[k],
                                    P_orb_k, df['inc'].values[k],
                                    a, df["R_s"].values[k], u1, u2,
                                    ecc_k, argp_k,
                                    fluxratios_comp[k], comp
                                    )
                        else:
                            best_model = np.ones(len(model_time))
                            sec_model = np.ones(
                                len(sec_model_time) if has_sec_data else 100)

                        # primary panel
                        y_fmt = ticker.ScalarFormatter(useOffset=False)
                        ax[i, col].yaxis.set_major_formatter(y_fmt)
                        ax[i, col].errorbar(
                            time, flux, flux_err_val, fmt=".", ms=3,
                            color=DATA_C, alpha=0.5, elinewidth=0.7, zorder=0,
                            rasterized=True)
                        ax[i, col].plot(
                            model_time, best_model, "-", color=PRI_C, lw=1.8,
                            zorder=2)
                        ax[i, col].annotate(
                            str(df["scenario"].values[k]) + " pri",
                            xy=(0.05, 0.06),
                            xycoords="axes fraction", fontsize=10)

                        # secondary panel: renormalise the secondary data into
                        # the host-star frame to match the (un-diluted) model.
                        y_fmt2 = ticker.ScalarFormatter(useOffset=False)
                        ax[i, sec_col].yaxis.set_major_formatter(y_fmt2)
                        if has_sec_data:
                            flux_sec_rn, err_sec_rn = renorm_flux(
                                flux_secondary, err_secondary,
                                self.stars["fluxratio"].values[star_idx]
                                )
                            ax[i, sec_col].errorbar(
                                time_secondary, flux_sec_rn, err_sec_rn,
                                fmt=".", ms=3, color=DATA_C, alpha=0.5,
                                elinewidth=0.7, zorder=0, rasterized=True)
                            ax[i, sec_col].plot(
                                sec_model_time, sec_model, "-", color=SEC_C,
                                lw=1.8, zorder=2)
                        else:
                            ax[i, sec_col].plot(
                                auto_sec_time, sec_model, "-", color=SEC_C,
                                lw=1.8, zorder=2)
                        ax[i, sec_col].annotate(
                            str(df["scenario"].values[k]) + " sec",
                            xy=(0.05, 0.06),
                            xycoords="axes fraction", fontsize=10)

        # column titles
        col_titles = ["TP", "EB primary", "EB secondary",
                       "EBx2P even" if has_evenodd else "EBx2P primary",
                       "EBx2P odd" if has_evenodd else "EBx2P secondary"]
        for c, title in enumerate(col_titles):
            ax[0, c].set_title(title, fontsize=12)

        # x-axis labels on bottom row
        for c in range(n_cols):
            ax[n_rows - 1, c].set_xlabel(
                "days from mid-transit", fontsize=10)
        # y-axis label on leftmost columns
        for i in range(n_rows):
            ax[i, 0].set_ylabel("normalized flux", fontsize=10)

        # Align the y-axis within each scenario row so its panels share one
        # depth scale, and clean up the panel frames.
        for i in range(n_rows):
            los = [ax[i, c].get_ylim()[0] for c in range(n_cols)]
            his = [ax[i, c].get_ylim()[1] for c in range(n_cols)]
            for c in range(n_cols):
                ax[i, c].set_ylim(min(los), max(his))
        for a_ in ax.flat:
            a_.spines[["top", "right"]].set_visible(False)
            a_.tick_params(labelsize=8)

        if save is False:
            plt.tight_layout()
            plt.show()
        elif (save is True) & (fname is None):
            plt.tight_layout()
            target_star = self.stars.ID.values[0]
            plt.savefig("TIC"+str(target_star)+"_fits.pdf")
        else:
            plt.tight_layout()
            plt.savefig(fname+".pdf")
        return
