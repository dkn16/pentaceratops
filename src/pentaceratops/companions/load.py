import os

from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from scipy import interpolate
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _resolve(path, default_name):
    """Resolve a data file path: absolute -> as-is; bare name -> package data dir."""
    if path is None:
        return os.path.join(_DATA_DIR, default_name)
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(_DATA_DIR, path)


def load_isochrones(age, model_path=None):
    # Logic: Read file, interpolate mass-to-magnitude for the specific target age.
    # Return: Interpolator function f(mass) -> Absolute G Mag
    model_path = _resolve(model_path, "BHAC15_CFHT.npy")

    data = np.load(model_path, allow_pickle=True).item()

    age_list = data.keys()

    age_keys = np.sort(list(age_list))

    # Guard against ages outside the isochrone grid. If the caller passes an
    # age < min or > max, we clip to the nearest available isochrone rather
    # than raising an IndexError on the empty slice.
    age_clipped = float(np.clip(age, age_keys[0], age_keys[-1]))
    age_lo = age_keys[age_keys <= age_clipped][-1]
    age_hi = age_keys[age_keys >= age_clipped][0]
    age = age_clipped

    # Linear interpolation between the two isochrones
    mass_lo, mag_lo = data[age_lo]['M/Ms'], data[age_lo]['G']
    mass_hi, mag_hi = data[age_hi]['M/Ms'], data[age_hi]['G']

    # Create interpolation functions for both ages
    f_lo = interpolate.interp1d(mass_lo, mag_lo, bounds_error=False, fill_value='extrapolate')
    f_hi = interpolate.interp1d(mass_hi, mag_hi, bounds_error=False, fill_value='extrapolate')

    def mass_to_mag(mass):
        mag1 = f_lo(mass)
        mag2 = f_hi(mass)
        # Interpolate between the two ages
        if age_hi == age_lo:
            return mag1
        weight = (age - age_lo) / (age_hi - age_lo)
        return mag1 * (1 - weight) + mag2 * weight

    return mass_to_mag


def load_ruwe_grid(parallax, grid_path=None):
    # Logic: Read the projected separation vs Delta G vs Expected RUWE table.
    # Return: 2D Interpolator function f(sep_au, delta_g) -> Predicted_RUWE
    grid_path = _resolve(grid_path, "RuweTableGP.txt")
    t = Table.read(grid_path, format='ascii', delimiter=' ')

    ruwe_dist = t

    star_distance = 1 / parallax  # in kpc
    star_distance *= 2.063e+8  # in au

    ruwe_dist['Sep(AU)'] = [star_distance * np.tan(np.radians(x/(3.6e6))) for x in np.power(10, ruwe_dist['log(sep)'])]
    sep_min = np.min(ruwe_dist['Sep(AU)'])
    sep_max = np.max(ruwe_dist['Sep(AU)'])

    # Pivot via pandas so the grid layout is independent of the row ordering
    # in the input file (the old reshape silently assumed a specific sort).
    ruwe_df = ruwe_dist.to_pandas()
    x_edges = np.sort(ruwe_df['Sep(AU)'].unique())
    y_edges = np.sort(ruwe_df['DeltaG'].unique())

    z = (ruwe_df.pivot_table(index='Sep(AU)', columns='DeltaG',
                             values='log(RUWE)')
               .reindex(index=x_edges, columns=y_edges)
               .values)
    z_sigma = (ruwe_df.pivot_table(index='Sep(AU)', columns='DeltaG',
                                   values='sigma_log(RUWE)')
                     .reindex(index=x_edges, columns=y_edges)
                     .values)

    values = np.stack((z, z_sigma), axis=-1)

    combined_interpolator = interpolate.RegularGridInterpolator(
        (x_edges, y_edges), values, bounds_error=False, fill_value=None
    )

    return combined_interpolator, sep_min, sep_max


def create_completeness_interpolator(parallax, filepath=None, detection_limit=18.0):
    """
    Creates a function f(distance, magnitude) -> completeness using Pandas for data processing.
    """
    filepath = _resolve(filepath, "gaia_contrast.txt")

    star_distance = 1 / parallax  # in kpc
    star_distance *= 2.063e+8  # in au

    df = pd.read_csv(
            filepath,
            sep=r'\s+',
            names=['Sep', '20%', '50%', '70%', '90%'],
            header=0,
            on_bad_lines='skip'
        )
    df = df.dropna(subset=['Sep'])
    df['Sep'] = df['Sep'].astype(float)

    df_melt = df.melt(id_vars='Sep', var_name='Completeness_Str', value_name='Mag')
    comp_map = {'20%': 0.2, '50%': 0.5, '70%': 0.7, '90%': 0.9}
    df_melt['Completeness'] = df_melt['Completeness_Str'].map(comp_map)
    df_melt['Mag'] = pd.to_numeric(df_melt['Mag'], errors='coerce')
    df_clean = df_melt.dropna(subset=['Sep', 'Mag', 'Completeness'])[['Sep', 'Mag', 'Completeness']]

    unique_seps = df_clean['Sep'].unique()
    mag_100_vals = np.where(unique_seps < 4000, 0.0, detection_limit)

    df_100 = pd.DataFrame({
        'Sep': unique_seps,
        'Mag': mag_100_vals,
        'Completeness': 1.0
    })

    df_final = pd.concat([df_clean, df_100], ignore_index=True)
    df_final['Sep'] = star_distance * np.tan(np.radians(df_final['Sep'].astype(float)/(3.6e6)))
    df_final = df_final.sort_values('Completeness')
    df_final = df_final.drop_duplicates(subset=['Sep', 'Mag'], keep='last')

    # Clip interpolator output to [0, 1] to guard against extrapolation outside
    # the convex hull producing non-physical completeness values.
    points = df_final[['Sep', 'Mag']].values
    raw_values = df_final['Completeness'].values
    _raw_interp = LinearNDInterpolator(points, raw_values, fill_value=0.0)

    def interpolator(pts):
        return np.clip(_raw_interp(pts), 0.0, 1.0)

    return interpolator, star_distance


def get_target_gaia_data(ra, dec):
    from astroquery.gaia import Gaia
    # Logic: Query Gaia DR3.
    # Return: Dictionary containing parallax, ruwe, phot_g_mean_mag, bp_rp
    coordinate = SkyCoord(ra, dec, frame='icrs', unit='deg')

    job_str = ("SELECT TOP 10 DISTANCE(POINT('ICRS', %f, %f), POINT('ICRS', ra, dec)) AS dist, * "
               "FROM gaiadr3.gaia_source "
               "WHERE 1=CONTAINS(POINT('ICRS', %f, %f),CIRCLE('ICRS', ra, dec, 0.08333333)) "
               "ORDER BY dist ASC" % (
                   coordinate.ra.degree, coordinate.dec.degree,
                   coordinate.ra.degree, coordinate.dec.degree))

    job = Gaia.launch_job(job_str)
    gaia_info = job.get_results()
    if gaia_info and len(gaia_info) >= 1:
        gmag = gaia_info['phot_g_mean_mag'][0]
        color = gaia_info['bp_rp'][0]
        n_good_obs = gaia_info['astrometric_n_good_obs_al'][0]
        astrometric_chi2 = gaia_info['astrometric_chi2_al'][0]
        parallax = gaia_info['parallax'][0]
        parallax_error = gaia_info['parallax_error'][0]
        ln_ruwe = np.log(gaia_info['ruwe'][0])

        if 3.6 <= gmag <= 21. and -1 <= color <= 10 and gaia_info['astrometric_params_solved'][0] == 31:
            return {
                'gmag': gmag,
                'color': color,
                'n_good_obs': n_good_obs,
                'astrometric_chi2': astrometric_chi2,
                'parallax': parallax,
                'parallax_error': parallax_error,
                'ln_ruwe': ln_ruwe
            }
        else:
            raise ValueError("Gaia data for the target is out of bounds or invalid.")
    else:
        raise ValueError("No Gaia source found at the given coordinates.")


def get_target_gaia_age(ra, dec, search_radius_deg=0.08333333 / 60):
    """Attempt to pull an age estimate for the target from Gaia DR3
    astrophysical_parameters (FLAME). Returns age in Gyr, or None if unavailable.
    """
    from astroquery.gaia import Gaia
    coordinate = SkyCoord(ra, dec, frame='icrs', unit='deg')
    job_str = (
        "SELECT TOP 5 "
        "DISTANCE(POINT('ICRS', %f, %f), POINT('ICRS', g.ra, g.dec)) AS dist, "
        "ap.age_flame "
        "FROM gaiadr3.gaia_source AS g "
        "JOIN gaiadr3.astrophysical_parameters AS ap "
        "ON g.source_id = ap.source_id "
        "WHERE 1=CONTAINS(POINT('ICRS', %f, %f), "
        "CIRCLE('ICRS', g.ra, g.dec, %f)) "
        "ORDER BY dist ASC" % (
            coordinate.ra.degree, coordinate.dec.degree,
            coordinate.ra.degree, coordinate.dec.degree,
            search_radius_deg,
        )
    )
    try:
        job = Gaia.launch_job(job_str)
        result = job.get_results()
    except Exception:
        return None
    if result is None or len(result) == 0:
        return None

    # `age_flame` is masked for the majority of DR3 sources — handle that
    # explicitly so a masked / None / NaN value always falls back cleanly.
    col = result['age_flame']
    mask = np.ma.getmaskarray(col) if np.ma.isMaskedArray(col) else np.zeros(len(col), dtype=bool)
    if mask[0]:
        return None
    age = col[0]
    try:
        age_f = float(age)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(age_f):
        return None
    return age_f
