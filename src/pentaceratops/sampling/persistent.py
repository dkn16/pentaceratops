import os
import numpy as np
from scipy.special import logsumexp
import copy
from .records import record_sampler

def compute_ess(log_weights):
    """
    Compute Effective Sample Size (ESS) from log-weights.
    """
    # Normalize log weights for stability
    log_w_norm = log_weights - logsumexp(log_weights)
    # ESS = 1 / sum(w^2) = exp( - logsumexp(2 * log_w_norm) )
    return np.exp(-logsumexp(2 * log_w_norm))

def solve_for_next_beta(current_beta, log_ls, log_S, target_ess, history_len):
    """
    Find the next beta such that the ESS of the persistent ensemble meets the target.
    
    Args:
        current_beta (float): The starting beta (lower bound).
        log_ls (np.array): Log likelihoods of all particles in pool.
        log_S (np.array): Log of the unnormalized mixture sum (sum_{s=0}^{t-1} p_s/Z_s).
        target_ess (float): Desired ESS.
        history_len (int): Number of past iterations (t), used for normalizing the mixture (1/t).
        
    Returns:
        tuple: (next_beta, log_weights_at_next_beta)
    """
    
    # We want to find beta > current_beta such that ESS(w) ~ target_ess
    # w_i = p_{new}(theta_i) / q_{mix}(theta_i)
    # In u-space, log_pi = 0.
    # log_w_i = (beta * log_L) - (log_S - log(t))
    
    log_t = np.log(history_len)
    
    # Define function to get log_weights for a given beta
    def get_log_weights(beta):
        log_numerator = beta * log_ls
        log_denominator = log_S - log_t
        return log_numerator - log_denominator

    # 1. Try stepping all the way to beta = 1.0
    beta_try = 1.0
    log_w_try = get_log_weights(beta_try)
    if compute_ess(log_w_try) >= target_ess:
        return 1.0, log_w_try

    # 2. Bisection search for beta
    low = current_beta
    high = 1.0
    best_beta = current_beta
    best_log_w = get_log_weights(current_beta)
    
    for _ in range(25): # 25 iterations is usually sufficient for high precision
        mid = 0.5 * (low + high)
        if mid <= low: # numerical limit
            break
            
        log_w = get_log_weights(mid)
        ess = compute_ess(log_w)
        
        if ess >= target_ess:
            # We can go higher
            best_beta = mid
            best_log_w = log_w
            low = mid
        else:
            # ESS < target
            high = mid
            
    return best_beta, best_log_w

def run_mcmc_uspace(seeds_u, seeds_log_ls, log_likelihood_function,
                     prior_transform, beta, n_steps, proposal_cov, initial_scale,
                     log_likelihood_function_batched=None,
                     prior_transform_batched=None):
    """
    Run Metropolis-Hastings MCMC in the Unit Cube (u-space) with adaptive scaling.
    
    Args:
        seeds_u: Starting positions in unit cube.
        seeds_log_ls: Cached log-likelihoods for the seeds.
        log_likelihood_function: Likelihood function.
        prior_transform: Transform from unit cube to parameters.
        beta: Inverse temperature.
        n_steps: Number of MCMC steps.
        proposal_cov: Base proposal covariance matrix (from weighted pool).
        initial_scale: Starting scale factor for the proposal.
        
    Returns:
        tuple: (new_us, new_log_ls, final_scale)
    """
    n_particles, ndim = seeds_u.shape
    current_us = seeds_u.copy()
    current_log_ls = seeds_log_ls.copy()
    
    current_scale = initial_scale
    
    # Target acceptance rate for optimal scaling
    target_accept = 0.234
    
    total_accepted = 0
    total_moves = 0
    
    for step in range(n_steps):
        # 1. Propose
        # Proposal = current + scale * Noise(0, Cov)
        noise = np.random.multivariate_normal(np.zeros(ndim), proposal_cov, size=n_particles)
        proposals_u = current_us + current_scale * noise
        
        # 2. Evaluate
        prop_log_ls = np.full(n_particles, -np.inf)
        
        # A. Bounds check (Unit Cube)
        in_bounds = np.all((proposals_u >= 0.0) & (proposals_u <= 1.0), axis=1)
        
        # B. Calculate Likelihoods for valid proposals
        valid_indices = np.where(in_bounds)[0]
        if (log_likelihood_function_batched is not None
                and prior_transform_batched is not None
                and len(valid_indices) > 0):
            # Vectorised path: one batched prior_transform + one batched lnL
            try:
                thetas_prop = prior_transform_batched(proposals_u[valid_indices])
                vals = np.asarray(log_likelihood_function_batched(thetas_prop),
                                  dtype=float)
                vals[~np.isfinite(vals)] = -np.inf
                prop_log_ls[valid_indices] = vals
            except Exception:
                # Fall back to per-particle scalar path
                for j, i in enumerate(valid_indices):
                    try:
                        theta_prop = prior_transform(proposals_u[i])
                        val = log_likelihood_function(theta_prop)
                        prop_log_ls[i] = val if not np.isnan(val) else -np.inf
                    except Exception:
                        prop_log_ls[i] = -np.inf
        elif (prior_transform_batched is not None and len(valid_indices) > 0):
            # Mixed path: batched prior_transform, scalar lnL. Saves the
            # per-call prior overhead (~hundreds of µs) without changing
            # the lnL evaluation path so results match the all-scalar mode
            # to numerical precision.
            try:
                thetas_prop = prior_transform_batched(proposals_u[valid_indices])
                for j, i in enumerate(valid_indices):
                    try:
                        val = log_likelihood_function(thetas_prop[j])
                        prop_log_ls[i] = val if not np.isnan(val) else -np.inf
                    except Exception:
                        prop_log_ls[i] = -np.inf
            except Exception:
                for j, i in enumerate(valid_indices):
                    try:
                        theta_prop = prior_transform(proposals_u[i])
                        val = log_likelihood_function(theta_prop)
                        prop_log_ls[i] = val if not np.isnan(val) else -np.inf
                    except Exception:
                        prop_log_ls[i] = -np.inf
        else:
            for i in valid_indices:
                try:
                    theta_prop = prior_transform(proposals_u[i])
                    val = log_likelihood_function(theta_prop)
                    prop_log_ls[i] = val if not np.isnan(val) else -np.inf
                except Exception:
                    prop_log_ls[i] = -np.inf

        # 3. Metropolis-Hastings Step
        # Log Ratio = beta * (log_L_prop - log_L_curr)
        # (Prior density is constant 1.0 in u-space, cancels out)
        log_alpha = beta * (prop_log_ls - current_log_ls)
        
        random_u = np.log(np.random.rand(n_particles))
        should_accept = in_bounds & (random_u < log_alpha)
        
        # Update state
        current_us[should_accept] = proposals_u[should_accept]
        current_log_ls[should_accept] = prop_log_ls[should_accept]
        
        # 4. Adaptation
        # Calculate average acceptance rate across all particles
        avg_accept_rate = np.mean(should_accept)
        
        # Diminishing adaptation factor based on LOCAL step
        # This resets every MCMC burst, allowing fast adaptation to the new beta
        gamma = 1.0 / ((step + 1) ** 0.5)
        
        # Update log_scale
        # log_sigma_{t+1} = log_sigma_t + gamma * (alpha_bar - alpha_star)
        log_scale = np.log(current_scale) + gamma * (avg_accept_rate - target_accept)
        current_scale = np.exp(log_scale)
        
        total_accepted += np.sum(should_accept)
        total_moves += n_particles

    return current_us, current_log_ls, current_scale

@record_sampler
def persistent_sampling(log_likelihood_function, prior_transform, n_active, ndim,
                        target_ess=None, mcmc_steps=20,
                        log_likelihood_function_batched=None,
                        prior_transform_batched=None):
    """
    A standalone implementation of Persistent Sampling (Karamanis & Seljak) 
    performing MCMC in the Unit Hypercube (u-space).

    Args:
        log_likelihood_function (callable): Function f(theta) -> log_likelihood.
        prior_transform (callable): Function f(u) -> theta, maps unit cube to physical space.
        n_active (int): Number of active particles (N) to generate per iteration.
        ndim (int): Dimensionality of the parameter space.
        target_ess (float, optional): Target ESS threshold. Defaults to 0.5 * total_particles (adaptive).
        mcmc_steps (int): Number of MCMC steps for the mutation phase.

    Returns:
        tuple: (log_evidence, weights, positions, log_likelihoods)
            - log_evidence: Estimate of log(Z) for the full model.
            - weights: Normalized importance weights of all particles for the posterior.
            - positions: Array of all particle positions (in theta space).
            - log_likelihoods: Array of all particle log likelihoods.
    """
    
    if target_ess is None:
        target_ess = 0.5 * n_active 

    # --- 1. Initialization (Iteration 0) ---
    # Draw valid samples from Prior (reject -inf likelihoods)
    # This handles pathological cases where parts of the prior have 0 likelihood (-inf logL)
    valid_us = []
    valid_thetas = []
    valid_log_ls = []
    
    # We must track how many attempts it takes to find valid samples
    # to correctly normalize the initial evidence Z_0 (Effective Prior Volume).
    total_attempts = 0

    # Initialization-draw policy (two stages):
    #   * VIABILITY GATE: if the first INIT_GATE draws yield *zero* finite
    #     likelihoods, the scenario has no usable prior region -> bail out and
    #     return lnZ = -inf. This kills truly-impossible scenarios cheaply.
    #   * COLLECT: once at least one valid particle has been found, keep drawing
    #     until the full n_active set is seeded, however many draws that takes
    #     (so low-valid-fraction scenarios like BEBx2P are still seeded fully).
    #     A generous absolute ceiling (MAX_INIT_TRIALS) backstops pathological
    #     near-zero-fraction cases so the loop can never run forever.
    # Both bounds are overridable via env vars.
    INIT_GATE = int(float(os.environ.get("PENTA_INIT_GATE", 1e6)))
    MAX_INIT_TRIALS = int(float(os.environ.get("PENTA_MAX_INIT_TRIALS", 5e7)))

    while len(valid_us) < n_active and total_attempts < MAX_INIT_TRIALS:
        # Viability gate: nothing finite after the first INIT_GATE draws -> stop.
        if len(valid_us) == 0 and total_attempts >= INIT_GATE:
            break
        n_needed = n_active - len(valid_us)
        n_batch = int(n_needed * 1.5) + 10
        batch_us = np.random.rand(n_batch, ndim)

        if (log_likelihood_function_batched is not None
                and prior_transform_batched is not None):
            try:
                batch_thetas = np.asarray(prior_transform_batched(batch_us))
                batch_lnls = np.asarray(
                    log_likelihood_function_batched(batch_thetas), dtype=float)
                for i in range(n_batch):
                    total_attempts += 1
                    val = batch_lnls[i]
                    if np.isfinite(val):
                        valid_us.append(batch_us[i])
                        valid_thetas.append(batch_thetas[i])
                        valid_log_ls.append(float(val))
                    if len(valid_us) >= n_active:
                        break
                continue
            except Exception:
                # Fall through to the per-particle path below
                pass

        # Mixed path: batched prior + scalar lnL
        if prior_transform_batched is not None:
            try:
                batch_thetas = np.asarray(prior_transform_batched(batch_us))
                for i in range(n_batch):
                    total_attempts += 1
                    val = log_likelihood_function(batch_thetas[i])
                    if np.isfinite(val):
                        valid_us.append(batch_us[i])
                        valid_thetas.append(batch_thetas[i])
                        valid_log_ls.append(float(val))
                    if len(valid_us) >= n_active:
                        break
                continue
            except Exception:
                pass

        for i in range(n_batch):
            total_attempts += 1
            u = batch_us[i]
            theta = prior_transform(u)
            val = log_likelihood_function(theta)
            if np.isfinite(val):
                valid_us.append(u)
                valid_thetas.append(theta)
                valid_log_ls.append(val)
            if len(valid_us) >= n_active:
                break
    
    # Viability gate failed (no finite likelihood in the first INIT_GATE draws):
    # the scenario's evidence is negligible, so return -inf and let it drop out
    # of the FPP instead of looping forever.
    if len(valid_us) == 0:
        print(f"PS init: no finite-likelihood prior sample in "
              f"{total_attempts} draws (gate {INIT_GATE}); returning lnZ = -inf.")
        # Return one dummy particle (not empty arrays) so downstream reductions
        # like np.max(log_likelihoods) and resampling don't crash; lnZ = -inf
        # makes the scenario drop out of the FPP regardless.
        return (-np.inf, np.array([1.0]), np.zeros((1, ndim)),
                np.array([-np.inf]))
    if len(valid_us) < n_active:
        print(f"PS init: only {len(valid_us)}/{n_active} valid particles after "
              f"{total_attempts} draws (hit absolute ceiling {MAX_INIT_TRIALS}); "
              f"proceeding with those.")
        n_active = len(valid_us)
        target_ess = min(target_ess, 0.5 * n_active)

    u_init = np.array(valid_us)
    thetas = np.array(valid_thetas)
    log_ls = np.array(valid_log_ls)

    # Storage for the Persistent Pool
    pool_us = u_init
    pool_thetas = thetas
    pool_log_ls = log_ls
    
    # History of schedule
    betas = [0.0]
    
    # Correct log Z_0 based on the proportion of valid prior samples
    # Z_0 = Integral(Prior * Indicator(L > 0)) ~= (N_valid / N_total) * Volume_Unit_Cube
    log_Z_0 = np.log(n_active) - np.log(total_attempts)
    log_Zs = [log_Z_0]
    
    # Initialize the "Mixture Sum" accumulator: log_S
    # S_0(u) = p_0(u)/Z_0 = Prior_u(u)/Z_0
    # In u-space, p_0(u) = 1.0 (for valid u). 
    # So log_S = log(1.0) - log(Z_0) = -log_Z_0
    pool_log_S = np.full(n_active, -log_Z_0, dtype=np.float64)
    
    # MCMC Adaptation State
    # Initial scale guess: 2.38 / sqrt(d) (assuming covariance is well estimated)
    current_scale = 2.38 / np.sqrt(ndim)
    
    current_beta = 0.0
    iteration = 0
    
    #print(f"Iter {iteration}: beta={current_beta:.4f}, log_Z={log_Z_0:.4f}, Pool Size={len(pool_us)}")

    # --- Main Loop ---
    while current_beta < 1.0:
        iteration += 1
        history_len = iteration 

        if iteration > 1e6:
            raise RuntimeError("Exceeded maximum trails in Persistent Sampling initialization.")
        
        # A. Find next temperature (beta)
        step_target_ess = target_ess 
        
        next_beta, log_weights = solve_for_next_beta(
            current_beta, pool_log_ls, pool_log_S, 
            step_target_ess, history_len
        )
        
        # B. Compute Evidence for this step (Z_t)
        current_log_Z = logsumexp(log_weights) - np.log(len(pool_us))
        
        betas.append(next_beta)
        log_Zs.append(current_log_Z)
        current_beta = next_beta
        
        #print(f"Iter {iteration}: beta={current_beta:.4f}, log_Z={current_log_Z:.4f}, Pool Size={len(pool_us)}, ESS={compute_ess(log_weights):.1f}, Scale={current_scale:.3f}")
        
        if current_beta >= 1.0:
            break
            
        # C. Update the Mixture Accumulator (log_S)
        # log_term = beta * log_L - log_Z (log_prior is 0 in u-space)
        log_term_t = (current_beta * pool_log_ls) - current_log_Z
        pool_log_S = np.logaddexp(pool_log_S, log_term_t)
        
        # D. Weighted Covariance Estimation
        # Normalize weights for covariance calculation
        # Weights w_i propto p_t / q_mix
        weights_norm = np.exp(log_weights - logsumexp(log_weights))
        
        # Compute Weighted Covariance of the ENTIRE pool in U-space
        # This gives us the shape of the current target distribution
        # np.cov returns a scalar for ndim=1; keep a square proposal matrix.
        cov_matrix = np.atleast_2d(np.cov(pool_us, rowvar=False, aweights=weights_norm))
        
        # Regularization (Nugget) to prevent singularity
        cov_matrix += np.eye(ndim) * 1e-6
        
        # Note: We do NOT multiply by (2.38^2/d) here.
        # We pass the raw covariance shape and handle scaling via `current_scale`
        # inside the MCMC runner.
        
        # E. Resample
        # Select N particles from the pool to be "active" seeds
        indices = np.random.choice(len(pool_us), size=n_active, p=weights_norm)
        seeds_u = pool_us[indices]
        
        # Optimization: Reuse cached likelihoods for seeds!
        seeds_log_ls = pool_log_ls[indices]
        
        # F. Mutate (Adaptive MCMC in U-Space)
        new_us, new_log_ls, current_scale = run_mcmc_uspace(
            seeds_u,
            seeds_log_ls,
            log_likelihood_function,
            prior_transform,
            current_beta,
            mcmc_steps,
            cov_matrix,
            current_scale,
            log_likelihood_function_batched=log_likelihood_function_batched,
            prior_transform_batched=prior_transform_batched,
        )

        # Convert new U to Theta for storage
        if prior_transform_batched is not None:
            try:
                new_thetas = np.asarray(prior_transform_batched(new_us))
            except Exception:
                new_thetas = np.array([prior_transform(u) for u in new_us])
        else:
            new_thetas = np.array([prior_transform(u) for u in new_us])
        
        # G. Add new particles to Persistent Pool
        # Update log_S for new particles
        beta_arr = np.array(betas) # shape (t+1,)
        log_Z_arr = np.array(log_Zs) # shape (t+1,)
        
        # (N_new, 1) * (1, T+1) -> (N_new, T+1)
        term_matrix = (new_log_ls[:, None] * beta_arr[None, :]) - log_Z_arr[None, :]
        new_log_S = logsumexp(term_matrix, axis=1)
        
        # Append to pool
        pool_us = np.vstack([pool_us, new_us])
        pool_thetas = np.vstack([pool_thetas, new_thetas])
        pool_log_ls = np.concatenate([pool_log_ls, new_log_ls])
        pool_log_S = np.concatenate([pool_log_S, new_log_S])
        
    # --- Finalization ---
    log_t_final = np.log(len(betas)) 
    log_numerators = 1.0 * pool_log_ls # log_prior is 0
    log_denominators = pool_log_S - log_t_final
    final_log_weights = log_numerators - log_denominators
    
    final_weights = np.exp(final_log_weights - logsumexp(final_log_weights))
    
    return log_Zs[-1], final_weights, pool_thetas, pool_log_ls

# --- Example Usage ---
if __name__ == "__main__":
    # Define a simple problem: 2D Gaussian
    ndim = 2
    true_mean = np.array([0.5, 0.5])
    cov_inv = np.eye(ndim) * 100.0 # sharp peak
    
    def log_likelihood(theta):
        diff = theta - true_mean
        return -0.5 * diff.T @ cov_inv @ diff

    def prior_transform(u):
        # Uniform [0, 1] -> [-2, 2]
        return 4.0 * u - 2.0

    print("Running Persistent Sampling on 2D Gaussian (U-space MCMC)...")
    log_Z, weights, positions, log_ls = persistent_sampling(
        log_likelihood_function=log_likelihood,
        prior_transform=prior_transform,
        n_active=100,
        ndim=ndim,
        target_ess=50,
        mcmc_steps=20
    )
    
    print("\nResults:")
    print(f"Estimated Log Evidence: {log_Z:.4f}")
    
    # Analytical Log Evidence
    sigma = 0.1
    log_z_analytical = np.log(2 * np.pi * sigma**2) - np.log(16.0)
    print(f"Analytical Log Evidence: {log_z_analytical:.4f}")
    
    # Weighted mean of position
    mean_est = np.average(positions, axis=0, weights=weights)
    print(f"Estimated Mean: {mean_est}")
