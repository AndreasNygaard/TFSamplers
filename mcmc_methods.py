import sys
import time
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tqdm import trange
from affine import affine_sample
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.get_logger().setLevel('ERROR')

from tools import LogProbCounter, trace_fn, MalaWithStepSize, MalaResults


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return "    Warning: " + str(msg) + '\n'
warnings.formatwarning = custom_formatwarning


### Affine Invariant Ensemble Sampler (AIES) ###

def run_affine(log_prob_fn,
               initial_state,
               n_steps=1000,
               progress_bar=True):

    if isinstance(initial_state, list):
        if len(initial_state) != 2:
            raise ValueError("If initial_state is a list, it must contain exactly two tensors.")
        if initial_state[0].shape != initial_state[1].shape:
            raise ValueError("If initial_state is a list, both tensors must have the same shape.")
        n_walkers = 2 * initial_state[0].shape[0]
    elif isinstance(initial_state, tf.Tensor):
        if len(initial_state.shape) != 2:
            raise ValueError("If initial_state is a tensor, it must have shape (n_walkers, n_params), where n_walkers is even.")
        elif initial_state.shape[0] % 2 != 0:
            raise ValueError("If initial_state is a tensor, the number of walkers (shape[0]) must be even.")
        n_walkers = initial_state.shape[0]
        initial_state = tf.split(initial_state, num_or_size_splits=2, axis=0)
    else:
        raise ValueError("initial_state must be either a tensor of shape (n_walkers, n_params) or a list of two tensors of shape (n_walkers/2, n_params), where n_walkers is even.")
    log_prob_counter = LogProbCounter(log_prob_fn)
    n_params = initial_state[0].shape[1]
    # run the sampler
    chain = affine_sample(log_prob_counter, n_steps, initial_state, args=[], progressbar=progress_bar)
    samples = tf.reshape(chain, [n_steps*n_walkers, n_params])
    acceptance_rate = tf.raw_ops.UniqueV2(x=samples, axis=[0])[0].shape[0]/samples.shape[0]
    n_evals = log_prob_counter.num_calls
    return samples, acceptance_rate, n_evals





### Hamiltonian Monte Carlo (HMC) ###

def run_hmc(log_prob_fn,
            initial_state,
            n_steps=100,
            covmat=None,
            num_leapfrog=5,
            step_size=0.1,
            num_adaptation_steps=None,
            num_burnin_steps=0,
            n_chains=10,
            use_diagonal_mass_matrix=False,
            progress_bar=True):

    if isinstance(initial_state, tf.Tensor):
        if len(initial_state.shape) == 1:
            initial_state = tf.repeat(tf.expand_dims(initial_state, axis=0), repeats=n_chains, axis=0)
        elif len(initial_state.shape) > 2:
            raise ValueError("If initial_state is a tensor, it must have shape (n_chains, n_params) or (n_params,).")
    else:
        raise ValueError("initial_state must be a tensor of shape (n_chains, n_params) or (n_params,).")

    n_chains = initial_state.shape[0]
    
    if covmat is None:
        scales = tf.math.reduce_std(initial_state, axis=0)
        if tf.reduce_any(scales == 0):
            covmat = tf.eye(initial_state.shape[1], dtype=tf.float32)
            warnings.warn("If covmat is not provided, it will be estimated from the initial state. However, if any parameter has zero variance across the initial chains, an identity covariance matrix will be used instead, which may lead to suboptimal performance. Consider providing a covariance matrix or ensuring that the initial state has non-zero variance across all parameters to mitigate this issue.")
        else:
            covmat = tf.power(scales,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    elif len(covmat.shape) < 2:
        covmat = tf.power(covmat,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    if use_diagonal_mass_matrix:
        covmat = tf.linalg.diag(tf.linalg.diag_part(covmat))
    L = tf.linalg.cholesky(covmat)

    def log_prob_whitened(z):
        x = initial_state + tf.linalg.matvec(L, z)
        log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        return log_prob_fn(x) + log_det

    log_prob_counter = LogProbCounter(log_prob_whitened)

    hcm_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_counter,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog
    )
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel = hcm_kernel,
        num_adaptation_steps=int(0.8 * num_burnin_steps) if num_adaptation_steps is None else num_adaptation_steps,
    )

    z0 = tf.zeros_like(initial_state, dtype=tf.float32)
    # Run the chain (with burn-in).
    @tf.function
    def run_chain():
        # Run the chain (with burn-in). 
        # Implements MCMC via repeated TransitionKernel steps.
        samples, trace = tfp.mcmc.sample_chain(
            num_results=n_steps+num_burnin_steps,
            num_burnin_steps=0,
            current_state=z0,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: trace_fn(_, pkr, n_steps, num_burnin_steps, pkr.inner_results, progress_bar=progress_bar)
        )
        samples = samples[num_burnin_steps:]
        return samples, trace

    samples, trace = run_chain()
    x_samples = initial_state + tf.linalg.matmul(samples, L, transpose_b=True)
    x_samples = tf.reshape(x_samples, [n_chains * n_steps, initial_state.shape[1]])
    acceptance_rate = tf.reduce_mean(tf.cast(trace[0], tf.float32))
    n_evals = log_prob_counter.num_calls
    return x_samples, acceptance_rate, n_evals



### No-U-Turn Sampler (NUTS) ###

def run_nuts(log_prob_fn,
             initial_state,
             n_steps=100,
             covmat=None,
             max_tree_depth=6,
             step_size=0.1,
             num_adaptation_steps=None,
             target_accept_prob=0.8,
             num_burnin_steps=0,
             n_chains=10,
             use_diagonal_mass_matrix=False,
             progress_bar=True):

    if isinstance(initial_state, tf.Tensor):
        if len(initial_state.shape) == 1:
            initial_state = tf.repeat(tf.expand_dims(initial_state, axis=0), repeats=n_chains, axis=0)
        elif len(initial_state.shape) > 2:
            raise ValueError("If initial_state is a tensor, it must have shape (n_chains, n_params) or (n_params,).")
    else:
        raise ValueError("initial_state must be a tensor of shape (n_chains, n_params) or (n_params,).")

    n_chains = initial_state.shape[0]

    if covmat is None:
        scales = tf.math.reduce_std(initial_state, axis=0)
        if tf.reduce_any(scales == 0):
            covmat = tf.eye(initial_state.shape[1], dtype=tf.float32)
            warnings.warn("If covmat is not provided, it will be estimated from the initial state. However, if any parameter has zero variance across the initial chains, an identity covariance matrix will be used instead, which may lead to suboptimal performance. Consider providing a covariance matrix or ensuring that the initial state has non-zero variance across all parameters to mitigate this issue.")
        else:
            covmat = tf.power(scales,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    elif len(covmat.shape) < 2:
        covmat = tf.power(covmat,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    if use_diagonal_mass_matrix:
        covmat = tf.linalg.diag(tf.linalg.diag_part(covmat))
    L = tf.linalg.cholesky(covmat)

    def log_prob_whitened(z):
        x = initial_state + tf.linalg.matvec(L, z)
        log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        return log_prob_fn(x) + log_det
    
    log_prob_counter = LogProbCounter(log_prob_whitened)

    nuts = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_prob_counter,
        step_size=step_size,
        max_tree_depth=max_tree_depth,
    )

    nuts = tfp.mcmc.DualAveragingStepSizeAdaptation(
        nuts,
        num_adaptation_steps=num_burnin_steps if num_adaptation_steps is None else num_adaptation_steps,
        target_accept_prob=target_accept_prob
    )

    z0 = tf.zeros_like(initial_state, dtype=tf.float32)

    # Run the chain (with burn-in).
    @tf.function
    def run_chain():
        # Run the chain (with burn-in). 
        # Implements MCMC via repeated TransitionKernel steps.
        samples, trace = tfp.mcmc.sample_chain(
            num_results=n_steps+num_burnin_steps,
            num_burnin_steps=0,
            current_state=z0,
            kernel=nuts,
            trace_fn=lambda _, pkr: trace_fn(_, pkr, n_steps, num_burnin_steps, pkr.inner_results, progress_bar=progress_bar)
            )
        samples = samples[num_burnin_steps:]
        return samples, trace

    samples, trace = run_chain()
    x_samples = initial_state + tf.linalg.matmul(samples, L, transpose_b=True)
    x_samples = tf.reshape(x_samples, [n_chains * n_steps, initial_state.shape[1]])
    acceptance_rate = tf.reduce_mean(tf.cast(trace[0], tf.float32))
    n_evals = log_prob_counter.num_calls
    return x_samples, acceptance_rate, n_evals




### Microcanonical Hamiltonian Monte Carlo (MCHMC) ###

@tf.function
def compute_energy_and_gradients(model, q):
    with tf.GradientTape() as tape:
        tape.watch(q)
        logp = model(q)
        energy = -logp
    grad = tape.gradient(energy, q)
    return energy, grad

@tf.function
def mchmc_step(model, q, p, dt=1e-3):
    """
    One microcanonical dynamics step
    q: position (parameter vector)
    p: momentum (same shape as q)
    """
    # half step p
    energy, grad = compute_energy_and_gradients(model, q)
    #tf.debugging.check_numerics(energy, "Energy is NaN")
    #tf.debugging.check_numerics(grad, "Grad is NaN")
    p_half = p - 0.5 * dt * grad
    
    # full step q
    q_new = q + dt * p_half

    # new gradient
    energy_new, grad_new = compute_energy_and_gradients(model, q_new)

    # complete p update
    p_new = p_half - 0.5 * dt * grad_new

    # compute new kinetic
    K_new = tf.reduce_sum(p_new * p_new) / 2.0
    desired_K = tf.reduce_sum(p * p) / 2.0
    scale = tf.sqrt(desired_K / (K_new + 1e-12))
    p_new = p_new * scale

    return q_new, p_new, dt*p_half


def kinetic_energy(p):
    return 0.5 * tf.reduce_sum(p * p, axis=-1)

def potential_energy(q, log_prob_fn):
    return -log_prob_fn(q)  # shape (num_chains,)

def hamiltonian(q, p, log_prob_fn):
    return potential_energy(q, log_prob_fn) + kinetic_energy(p)

def run_mchmc(log_prob_fn,
              initial_state,
              n_steps=100,
              scales=None,
              num_leapfrog=10,
              step_size=0.1,
              n_chains=50,
              progress_bar=True):

    if isinstance(initial_state, tf.Tensor):
        if len(initial_state.shape) == 1:
            initial_state = tf.repeat(tf.expand_dims(initial_state, axis=0), repeats=n_chains, axis=0)
        elif len(initial_state.shape) > 2:
            raise ValueError("If initial_state is a tensor, it must have shape (n_chains, n_params) or (n_params,).")
    else:
        raise ValueError("initial_state must be a tensor of shape (n_chains, n_params) or (n_params,).")

    n_chains = initial_state.shape[0]

    if scales is None:
        scales = tf.math.reduce_std(initial_state, axis=0)
        if tf.reduce_any(scales == 0):
            raise ValueError("If scales are not provided, they will be estimated from the initial state. However, if any parameter has zero variance across the initial chains, this will lead to zero scales and thus NaNs in the leapfrog updates. Please provide non-zero scales for all parameters either as a tensor of shape (n_params,) or as a scalar which will be broadcasted to all parameters, or ensure that the initial state has non-zero variance across all parameters.")

    dim = initial_state.shape[1]
    n_chains = initial_state.shape[0]

    q = initial_state

    samples = []
    acceptance_count = tf.zeros(n_chains, dtype=tf.float32)
    log_prob_counter = LogProbCounter(log_prob_fn)

    avg_num_leapfrog = 0.0
    count_nan_encountered = 0

    loop_fn = trange if progress_bar else range
    for i in loop_fn(n_steps):
        if not progress_bar:
            print("Step", i+1, "of", n_steps, " "*8, end='\r')

        # --- Sample fresh momentum ---
        p = tf.random.normal([n_chains, dim])

        # Save initial state
        q0 = q
        p0 = p

        # Initial Hamiltonian
        H0 = hamiltonian(q0, p0, log_prob_counter)

        # --- Leapfrog trajectory ---
        q_prop = q0
        p_prop = p0

        for t in range(num_leapfrog):
            q_prop_old = q_prop
            p_prop_old = p_prop
            q_prop, p_prop, _ = mchmc_step(
                log_prob_counter,
                q_prop,
                p_prop,
                dt=step_size*scales
            )
            if tf.reduce_any(tf.math.is_nan(q_prop)) or tf.reduce_any(tf.math.is_nan(p_prop)):
                avg_num_leapfrog = (avg_num_leapfrog * count_nan_encountered + t) / (count_nan_encountered + 1)
                count_nan_encountered += 1
                # Revert to previous state
                q_prop = q_prop_old
                p_prop = p_prop_old
                break

        # Negate momentum for reversibility
        p_prop = -p_prop

        # Proposed Hamiltonian
        H_prop = hamiltonian(q_prop, p_prop, log_prob_counter)

        # --- Metropolis correction ---
        log_accept_ratio = -(H_prop - H0)

        u = tf.math.log(tf.random.uniform([n_chains]))
        accept = u < log_accept_ratio

        accept = tf.cast(accept, tf.float32)

        # Update positions (reject -> keep old q)
        q = tf.where(tf.expand_dims(accept > 0, -1), q_prop, q0)

        acceptance_count += accept

        samples.append(q)
    if not progress_bar:
        print()

    avg_num_leapfrog = (avg_num_leapfrog * count_nan_encountered + num_leapfrog * (n_steps - count_nan_encountered)) / n_steps

    ratio_num_leapfrog = avg_num_leapfrog / num_leapfrog
    if ratio_num_leapfrog < 0.6:
        warnings.warn(f"NaNs were encountered significantly often during leapfrog integration in MCHMC. On average, only {avg_num_leapfrog:.2f} out of {num_leapfrog} leapfrog steps were completed before NaNs were encountered. This may indicate that the step size is too large or that the target distribution has regions of very high curvature. Consider reducing the step size or initialising the sampler in a region of higher probability to mitigate this issue.")

    samples = tf.stack(samples, axis=0)
    samples = tf.reshape(samples, [n_chains * n_steps, dim])
    acceptance_rate = acceptance_count / n_steps
    n_evals = log_prob_counter.num_calls
    return samples, acceptance_rate, n_evals




### Metropolis Adjusted Langevin Algorithm (MALA) ###

def run_mala(log_prob_fn,
             initial_state,
             n_steps=100,
             covmat=None,
             step_size=0.01,
             num_burnin_steps=1000,
             num_steps_between_results=0,
             volatility_fn=None,
             n_chains=50,
             use_diagonal_mass_matrix=False,
             progress_bar=True):

    if isinstance(initial_state, tf.Tensor):
        if len(initial_state.shape) == 1:
            initial_state = tf.repeat(tf.expand_dims(initial_state, axis=0), repeats=n_chains, axis=0)
        elif len(initial_state.shape) > 2:
            raise ValueError("If initial_state is a tensor, it must have shape (n_chains, n_params) or (n_params,).")
    else:
        raise ValueError("initial_state must be a tensor of shape (n_chains, n_params) or (n_params,).")

    n_chains = initial_state.shape[0]

    if covmat is None:
        scales = tf.math.reduce_std(initial_state, axis=0)
        if tf.reduce_any(scales == 0):
            covmat = tf.eye(initial_state.shape[1], dtype=tf.float32)
            warnings.warn("If covmat is not provided, it will be estimated from the initial state. However, if any parameter has zero variance across the initial chains, an identity covariance matrix will be used instead, which may lead to suboptimal performance. Consider providing a covariance matrix or ensuring that the initial state has non-zero variance across all parameters to mitigate this issue.")
        else:
            covmat = tf.power(scales,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    elif len(covmat.shape) < 2:
        covmat = tf.power(covmat,2) * tf.eye(initial_state.shape[1], dtype=tf.float32)
    if use_diagonal_mass_matrix:
        covmat = tf.linalg.diag(tf.linalg.diag_part(covmat))
    L = tf.linalg.cholesky(covmat)

    def log_prob_whitened(z):
        x = initial_state + tf.linalg.matvec(L, z)
        log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        return log_prob_fn(x) + log_det

    log_prob_counter = LogProbCounter(log_prob_whitened)
    
    if volatility_fn is None:
        def volatility_fn(x):
            return 1. / (0.5 + 0.1 * tf.math.abs(x))

    z0 = tf.zeros_like(initial_state, dtype=tf.float32)

    def mala_step_size_getter_fn(kernel_results):
        return kernel_results.step_size


    def mala_step_size_setter_fn(kernel_results, new_step_size):
        return MalaResults(
            inner_results=kernel_results.inner_results,
            step_size=new_step_size
        )

    wrapped_mala = MalaWithStepSize(
        target_log_prob_fn=log_prob_counter,
        step_size=step_size,
        volatility_fn=volatility_fn
    )
    adaptive_mala = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=wrapped_mala,
        num_adaptation_steps=int(1.0 * num_burnin_steps),
        target_accept_prob=0.574,
        step_size_getter_fn=mala_step_size_getter_fn,
        step_size_setter_fn=mala_step_size_setter_fn,
    )

    @tf.function
    def run_chain():
        samples, trace = tfp.mcmc.sample_chain(
            num_results=n_steps+num_burnin_steps,
            current_state=z0,
            kernel=adaptive_mala,
            num_burnin_steps=0,
            num_steps_between_results=num_steps_between_results,
            trace_fn=lambda _, pkr: trace_fn(_,
                                             pkr,
                                             n_steps,
                                             num_burnin_steps,
                                             pkr.inner_results.inner_results,
                                             num_steps_between_results=num_steps_between_results,
                                             progress_bar=progress_bar),
            seed=42)
        samples = samples[num_burnin_steps:]
        return samples, trace

    samples, trace = run_chain()
    x_samples = initial_state + tf.linalg.matmul(samples, L, transpose_b=True)
    x_samples = tf.reshape(x_samples, [n_chains * n_steps, initial_state.shape[1]])
    acceptance_rate = tf.reduce_mean(tf.cast(trace[0], tf.float32)).numpy()
    n_evals = log_prob_counter.num_calls.numpy()
    return x_samples, acceptance_rate, n_evals
