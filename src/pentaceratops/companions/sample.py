import numpy as np
from scipy import optimize
import scipy.stats as stats

from .load import (
    load_isochrones,
    load_ruwe_grid,
    create_completeness_interpolator,
    get_target_gaia_data,
)


def calculate_projected_separation(a, e, cos_i, arg_peri, period, phase):
    # `phase` is sampled uniformly on [0, 2pi] and represents the mean anomaly
    # at the observation epoch directly; no absolute time offset is needed.
    M = phase

    def kepler_eqn(E):
        return E - e * np.sin(E) - M

    def kepler_prime(E):
        return 1 - e * np.cos(E)

    current_E, convergence, _ = optimize.newton(kepler_eqn, M, fprime=kepler_prime, full_output=True, maxiter=100)
    f = 2 * np.arctan2(np.tan(current_E / 2), np.sqrt((1 - e) / (1 + e)))
    alpha = f + arg_peri
    sqtmp = np.sqrt(np.sin(alpha)**2 + np.cos(alpha)**2 * cos_i**2)
    pro_sep = a * (1-e**2)/(1+e*np.cos(f))*sqtmp  # AU
    return pro_sep, convergence


def calculate_contrast(primary_mass, mass_ratios, isochrone_interpolator):
    primary_mag = isochrone_interpolator(primary_mass)
    companion_mag = isochrone_interpolator(primary_mass * mass_ratios)
    delta_g = np.subtract(companion_mag, primary_mag)
    return delta_g, primary_mag


def sample_periods(n_samples):
    mu = 5.03
    sigma = 2.28
    log_periods = np.random.normal(mu, sigma, n_samples)
    periods = 10**log_periods
    periods = np.clip(periods, 0.1, np.inf)
    return periods


def sample_eccentricities(periods, n_samples):
    u = np.random.rand(n_samples)
    mu_e = np.log10(periods)*0.148 + 0.001
    sigma_e = np.log10(periods)*0.042 + 0.128

    lower_bound_ecc = stats.norm.cdf(0, loc=mu_e, scale=sigma_e)
    upper_bound_ecc = stats.norm.cdf(1, loc=mu_e, scale=sigma_e)

    eccs = stats.norm.ppf(u*(upper_bound_ecc - lower_bound_ecc) + lower_bound_ecc,
                          loc=mu_e, scale=sigma_e)
    eccs = np.clip(eccs, 0, 0.9999)
    return eccs


def sample_mass_ratios(n_samples):
    return np.random.uniform(0, 1, n_samples)


def sample_inclinations(n_samples):
    cos_i = np.random.uniform(0, 1, n_samples)
    return np.arccos(cos_i)


def sample_arguments_of_periapsis(n_samples):
    return np.random.uniform(0, 2 * np.pi, n_samples)


def sample_phase_of_periapsis(n_samples):
    return np.random.uniform(0, 2 * np.pi, n_samples)


def sample_priors(n_samples):
    samples = {}
    samples['periods'] = sample_periods(n_samples)
    samples['eccentricities'] = sample_eccentricities(samples['periods'], n_samples)
    samples['mass_ratios'] = sample_mass_ratios(n_samples)
    samples['inclinations'] = sample_inclinations(n_samples)
    samples['arguments_of_periapsis'] = sample_arguments_of_periapsis(n_samples)
    samples['pericenter_phase'] = sample_phase_of_periapsis(n_samples)
    return samples


def run_ruwe_rejection(sim_population, observed_ruwe, ruwe_interpolator, sep_min, sep_max, convergence):
    pred_log_ruwe, pred_sigma = ruwe_interpolator(np.array([sim_population['proj_sep_au'],
                                                  sim_population['delta_g']]).T).T

    # `pred_sigma` is the std of log10(RUWE); propagate to linear space via
    # sigma_lin ~= ln(10) * 10**pred_log_ruwe * pred_sigma (delta method).
    pred_ruwe_lin = 10.0**pred_log_ruwe
    sigma_lin = np.log(10.0) * pred_ruwe_lin * pred_sigma
    sigma_lin = np.where(np.isfinite(sigma_lin) & (sigma_lin > 0), sigma_lin, np.inf)
    probs = stats.halfnorm.cdf(pred_ruwe_lin, loc=observed_ruwe, scale=sigma_lin)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    bound_delta_g = (sim_population['delta_g'] > -0.1) & (sim_population['delta_g'] < 7.0)
    bound_proj_sep = (sim_population['proj_sep_au'] > sep_min) & (sim_population['proj_sep_au'] < sep_max)

    probs[bound_delta_g == False] = 0.0
    probs[bound_proj_sep == False] = 0.0

    random_draws = np.random.rand(len(probs))
    is_rejected = random_draws < probs
    is_rejected = is_rejected | (convergence == False)
    return is_rejected


def run_gaia_limit_rejection(sim_population, completeness_interpolator, four_arc, completeness_mag):
    prob = completeness_interpolator(np.array([sim_population['proj_sep_au'],
                                              sim_population['delta_g_gaia']]).T)
    mask = sim_population['proj_sep_au'] > four_arc
    prob[mask] = np.where(sim_population['delta_g_gaia'][mask] < completeness_mag, 1.0, 0.0)

    random_draws = np.random.rand(len(prob))
    is_rejected = random_draws < prob
    return is_rejected


def stellar_companion(ra, dec, primary_mass, star_age, n_sim=100000):
    isochrone_interpolator = load_isochrones(star_age)
    isochrone_interpolator_gaia = load_isochrones(star_age, model_path='BHAC15_GAIA.npy')

    gaia_data = get_target_gaia_data(ra, dec)
    ruwe_interpolator, sep_min, sep_max = load_ruwe_grid(gaia_data['parallax'])
    completeness_interpolator, star_distance = create_completeness_interpolator(gaia_data['parallax'])

    samples = sample_priors(n_sim)
    semi_major_axes = ((samples['periods'] / 365)**2 * 39.478 * primary_mass*(1 + samples['mass_ratios'])/(4 * np.pi ** 2))**(1/3)

    samples['proj_sep_au'], convergence = calculate_projected_separation(
        semi_major_axes,
        samples['eccentricities'],
        np.cos(samples['inclinations']),
        samples['arguments_of_periapsis'],
        samples['periods'],
        samples['pericenter_phase']
    )

    samples['delta_g'], _ = calculate_contrast(primary_mass, samples['mass_ratios'], isochrone_interpolator)
    samples['delta_g_gaia'], primary_mag = calculate_contrast(primary_mass, samples['mass_ratios'], isochrone_interpolator_gaia)

    four_arc = (star_distance * 0.0000193906)
    completness_absolute = 18 - 5 * np.log10(star_distance / 2062650)
    completeness_mag = (completness_absolute - primary_mag)

    is_rejected_ruwe = run_ruwe_rejection(
        samples, np.exp(gaia_data['ln_ruwe']), ruwe_interpolator, sep_min, sep_max, convergence
    )
    is_rejected_gaia = run_gaia_limit_rejection(samples, completeness_interpolator, four_arc, completeness_mag)

    is_rejected = is_rejected_ruwe | is_rejected_gaia

    # Predicted RUWE for output (linear scale)
    pred_log_ruwe, _ = ruwe_interpolator(np.array([samples['proj_sep_au'],
                                                   samples['delta_g']]).T).T
    pred_ruwe = 10.0**pred_log_ruwe

    return {
        'mass_ratios': samples['mass_ratios'][~is_rejected],
        'eccentricities': samples['eccentricities'][~is_rejected],
        'arguments_of_periapsis': samples['arguments_of_periapsis'][~is_rejected],
        'inclinations': samples['inclinations'][~is_rejected],
        'periods': samples['periods'][~is_rejected],
        'pericenter_phase': samples['pericenter_phase'][~is_rejected],
        'semi_major_axes': semi_major_axes[~is_rejected],
        'proj_sep_au': samples['proj_sep_au'][~is_rejected],
        'delta_g': samples['delta_g'][~is_rejected],
        'delta_g_gaia': samples['delta_g_gaia'][~is_rejected],
        'predicted_ruwe': pred_ruwe[~is_rejected],
        'n_total': n_sim,
        'n_accepted': int(np.sum(~is_rejected)),
    }
