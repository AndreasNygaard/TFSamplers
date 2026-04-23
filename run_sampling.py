import warnings

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from hypersphere_sampler import HypersphereSampler

from mcmc_methods import run_mh, run_affine, run_hmc, run_nuts, run_mala

class SamplerResults():
    def __init__(self, samples, acceptance_rate, evaluations):
        self.samples = samples
        self.acceptance_rate = acceptance_rate
        self.evaluations = evaluations
        self.burnin_samples = None
        self.burnin_acceptance_rates = None
        self.burnin_evaluations = None
        self.covmat_estimate = None

    def set_burnin_results(self, burnin_samples, burnin_acceptance_rates, burnin_evaluations, covmat_estimate):
        self.burnin_samples = burnin_samples
        self.burnin_acceptance_rates = burnin_acceptance_rates
        self.burnin_evaluations = burnin_evaluations
        self.covmat_estimate = covmat_estimate

class Sampler:
    def __init__(self, log_prob_fn, bounds=None, enforce_boundaries=True, covmat=None, initial_state=None, n_chains=10, initial_distribution='repeat'):
        self.log_prob_no_bounds = log_prob_fn
        if bounds is not None:
            self.lower_bounds = bounds[0]
            self.upper_bounds = bounds[1]
        else:
            self.lower_bounds = None
            self.upper_bounds = None

        self.initial_state = None
        if initial_state is not None:
            self.set_initial_state(initial_state, n_chains=n_chains, initial_distribution=initial_distribution, bounds=(lower_bounds, upper_bounds))
            
        if enforce_boundaries and bounds is not None:
            self.log_prob_fn = self.create_bounded_log_prob_fn(self.log_prob_no_bounds)
        else:
            self.log_prob_fn = log_prob_fn

        if covmat is not None:
            if isinstance(covmat, list):
                covmat = tf.convert_to_tensor(covmat, dtype=tf.float32)
            if len(covmat.shape) == 1:
                self.ini_covmat = tf.linalg.diag(covmat**2)
            elif len(covmat.shape) == 2:
                self.ini_covmat = covmat
            else:
                raise ValueError("Covariance matrix must be either a 1D array (diagonal) or a 2D array (full covariance).")
        elif bounds is not None:
            self.ini_covmat = tf.linalg.diag((self.upper_bounds - self.lower_bounds) / 1000.0)**2
        else:
            warnings.warn("No covariance matrix provided and bounds not set.")


    def sample(self,
               initial_state=None,
               n_steps=100,
               method='affine',
               n_chains=10,
               initial_distribution='repeat',
               bounds=None,
               num_burnin_steps=100,
               num_covmat_updates=3,
               update_initial_state=True,
               update_initial_distribution=True,
               sampler_kwargs={},
               burnin_kwargs={},
               get_individual_chains=False):

        if num_covmat_updates > 0 and num_burnin_steps == 0:
            raise ValueError("Burn-in steps must be greater than 0 if covariance matrix updates are requested.")

        if initial_state is not None or initial_distribution == 'uniform':
            self.set_initial_state(initial_state, n_chains=n_chains, initial_distribution=initial_distribution, bounds=bounds)
        elif self.initial_state is None:
            raise ValueError("Initial state must be provided either during initialization, when calling sample(), or using the set_initial_state method.")
        sampler_kwargs.update({'get_individual_chains': get_individual_chains})

        if method == 'mh':
            sample_fn = lambda initial_state, steps, covmat, sampler_kwargs: run_mh(self.log_prob_fn,
                                                                                    initial_state,
                                                                                    n_steps=steps,
                                                                                    covmat=covmat,
                                                                                    **sampler_kwargs)
        elif method == 'affine':
            num_burnin_steps = 0  # Affine-invariant sampler doesn't use burn-in
            num_covmat_updates = 0  # Affine-invariant sampler doesn't update covariance
            sample_fn = lambda initial_state, steps, covmat, sampler_kwargs: run_affine(self.log_prob_fn,
                                                                                        initial_state,
                                                                                        n_steps=steps,
                                                                                        **sampler_kwargs)
        elif method == 'hmc':
            sample_fn = lambda initial_state, steps, covmat, sampler_kwargs: run_hmc(self.log_prob_fn,
                                                                                     initial_state,
                                                                                     n_steps=steps,
                                                                                     covmat=covmat,
                                                                                     **sampler_kwargs)
        elif method == 'nuts':
            sample_fn = lambda initial_state, steps, covmat, sampler_kwargs: run_nuts(self.log_prob_fn,
                                                                                      initial_state,
                                                                                      n_steps=steps,
                                                                                      covmat=covmat,
                                                                                      **sampler_kwargs)
        elif method == 'mala':
            sample_fn = lambda initial_state, steps, covmat, sampler_kwargs: run_mala(self.log_prob_fn,
                                                                                      initial_state,
                                                                                      n_steps=steps,
                                                                                      covmat=covmat,
                                                                                      **sampler_kwargs)
        else:
            raise ValueError("Invalid sampling method. Must be 'affine', 'hmc', 'nuts', 'mchmc', or 'mala'.")
        covmat_estimate = self.ini_covmat
        burnin_sampler_kwargs = sampler_kwargs.copy()
        burnin_sampler_kwargs.update(burnin_kwargs)
        burnin_samples = []
        burnin_acceptance_rates = []
        burnin_evaluations = []
        for i in range(num_covmat_updates):
            print(f"Estimatingcovariance matrix, iteration {i+1}/{num_covmat_updates}...")
            samples, acceptance_rate, evaluations = sample_fn(self.initial_state, num_burnin_steps, covmat_estimate, burnin_sampler_kwargs)
            burnin_samples.append(samples)
            burnin_acceptance_rates.append(acceptance_rate)
            burnin_evaluations.append(evaluations)
            covmat_estimate = tfp.stats.covariance(samples)
            L = tf.linalg.cholesky(covmat_estimate)
            covmat_estimate = tf.matmul(L, L, transpose_b=True)  # Ensure covariance matrix is positive definite
            bestfit_estimate = tf.reduce_mean(samples, axis=0)
            if tf.math.reduce_any(tf.math.is_nan(covmat_estimate)) or tf.math.reduce_any(tf.math.is_inf(covmat_estimate)):
                raise ValueError("Covariance matrix estimate contains NaNs or Infs. Use an initial state closer to the mode or use method='affine' instead to get a better initial state and covariance estimate.")

            if update_initial_state and initial_distribution != 'uniform' and len(initial_state.shape) == 1:
                if update_initial_distribution:
                    initial_distribution = 'gaussian'
                self.set_initial_state(bestfit_estimate,
                                       n_chains=n_chains,
                                       initial_distribution=initial_distribution,
                                       bounds=bounds,
                                       covmat=covmat_estimate)

        print("Running final sampling...")
        samples, acceptance_rate, evaluations = sample_fn(self.initial_state, n_steps, covmat_estimate, sampler_kwargs)

        sampler_results = SamplerResults(samples, acceptance_rate, evaluations)
        if num_covmat_updates > 0:
            sampler_results.set_burnin_results(burnin_samples, burnin_acceptance_rates, burnin_evaluations, covmat_estimate)

        return sampler_results
        


    def create_bounded_log_prob_fn(self, log_prob_fn):
        scales = self.upper_bounds - self.lower_bounds
        def box_log_prob(x):
            factor = 10000/scales
            below = factor*tf.exp(self.lower_bounds - x)      # penalty if below lower bound
            above = factor*tf.exp(x - self.upper_bounds)      # penalty if above upper bound
            inside = tf.zeros_like(x)                         # inside the box: uniform
            log_prob = tf.where(x < self.lower_bounds, -below,
                                tf.where(x > self.upper_bounds, -above, inside))
            return tf.reduce_sum(log_prob, axis=-1)

        @tf.function(reduce_retracing=True)
        def bounded_log_prob_fn(x):
            log_like = log_prob_fn(x)
            return log_like + box_log_prob(x)

        return bounded_log_prob_fn

    def set_covmat(self, covmat):
        if isinstance(covmat, list):
            covmat = tf.convert_to_tensor(covmat, dtype=tf.float32)
        if len(covmat.shape) == 1:
            self.ini_covmat = tf.linalg.diag(covmat)
        elif len(covmat.shape) == 2:
            self.ini_covmat = covmat
        else:
            raise ValueError("Covariance matrix must be either a 1D array (diagonal) or a 2D array (full covariance).")

    def set_bounds(self, lower_bounds, upper_bounds, overwrite_covmat=False, overwrite_log_prob_fn=True):
        if isinstance(lower_bounds, list):
            lower_bounds = tf.convert_to_tensor(lower_bounds, dtype=tf.float32)
        if isinstance(upper_bounds, list):
            upper_bounds = tf.convert_to_tensor(upper_bounds, dtype=tf.float32)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        if overwrite_covmat:
            self.ini_covmat = tf.linalg.diag((self.upper_bounds - self.lower_bounds) / 100.0)**2
        if overwrite_log_prob_fn:
            self.log_prob_fn = self.create_bounded_log_prob_fn(self.log_prob_no_bounds)

    def set_initial_state(self, initial_state, n_chains=10, initial_distribution='repeat', bounds=None, covmat=None):
        if isinstance(initial_state, list) and len(initial_state) > 2:
            initial_state = tf.convert_to_tensor(initial_state, dtype=tf.float32)
        elif isinstance(initial_state, np.ndarray) and len(initial_state.shape) > 2:
            initial_state = tf.convert_to_tensor(initial_state, dtype=tf.float32)
        elif isinstance(initial_state, list) and len(initial_state) == 2:
            if initial_state[0].shape != initial_state[1].shape:
                raise ValueError("If initial_state is a list of two tensors, both tensors must have the same shape.")

        if isinstance(initial_state, list) or (isinstance(initial_state, tf.Tensor) and len(initial_state.shape) == 2):
            pass
        elif initial_state is None and initial_distribution == 'uniform':
            if bounds is not None:
                lower_bounds, upper_bounds = bounds
                if instance(lower_bounds, list):
                    lower_bounds = tf.convert_to_tensor(lower_bounds, dtype=tf.float32)
                elif instance(lower_bounds, np.ndarray):
                    lower_bounds = tf.convert_to_tensor(lower_bounds, dtype=tf.float32)
                if instance(upper_bounds, list):
                    upper_bounds = tf.convert_to_tensor(upper_bounds, dtype=tf.float32)
                elif instance(upper_bounds, np.ndarray):
                    upper_bounds = tf.convert_to_tensor(upper_bounds, dtype=tf.float32)
            elif self.lower_bounds is not None and self.upper_bounds is not None:
                lower_bounds = self.lower_bounds
                upper_bounds = self.upper_bounds
            else:
                raise ValueError("Bounds must be provided to initialize walkers uniformly.")
            initial_state = tf.random.uniform((n_chains, lower_bounds.shape[0]), minval=lower_bounds, maxval=upper_bounds)
        elif len(initial_state.shape) > 2:
            raise ValueError("Initial state must be either a 1D array (single point), a 2D array (multiple points).")
        elif len(initial_state.shape) == 1:
            if initial_distribution == 'repeat':
                initial_state = tf.tile(tf.expand_dims(initial_state, axis=0), (n_chains, 1))
            elif initial_distribution == 'gaussian':
                if bounds is not None:
                    if isinstance(bounds, list):
                        if instance(bounds[0], tf.Tensor):
                            bounds = np.array([bounds[0].numpy(), bounds[1].numpy()]).T
                        else:
                            bounds = np.array(bounds).T
                    elif isinstance(bounds, tf.Tensor):
                        bounds = np.array(bounds.numpy()).T
                    else:
                        bounds = np.array(bounds).T
                elif self.lower_bounds is not None and self.upper_bounds is not None:
                    bounds = np.array([self.lower_bounds.numpy(), self.upper_bounds.numpy()]).T
                else:
                    bounds = None
                if covmat is None:
                    covmat = self.ini_covmat
                dist = HypersphereSampler(initial_state.shape[0],
                                          limits=bounds,
                                          covmat=covmat.numpy(),
                                          centers=initial_state.numpy())
                initial_state = tf.convert_to_tensor(dist.sample(n_chains), dtype=tf.float32)
            elif initial_distribution == 'uniform':
                if bounds is not None:
                    lower_bounds, upper_bounds = bounds
                    if instance(lower_bounds, list):
                        lower_bounds = tf.convert_to_tensor(lower_bounds, dtype=tf.float32)
                    elif instance(lower_bounds, np.ndarray):
                        lower_bounds = tf.convert_to_tensor(lower_bounds, dtype=tf.float32)
                    if instance(upper_bounds, list):
                        upper_bounds = tf.convert_to_tensor(upper_bounds, dtype=tf.float32)
                    elif instance(upper_bounds, np.ndarray):
                        upper_bounds = tf.convert_to_tensor(upper_bounds, dtype=tf.float32)
                elif self.lower_bounds is not None and self.upper_bounds is not None:
                    lower_bounds = self.lower_bounds
                    upper_bounds = self.upper_bounds
                else:
                    raise ValueError("Bounds must be provided to initialize walkers uniformly.")
                initial_state = tf.random.uniform((n_chains, initial_state.shape[0]), minval=lower_bounds, maxval=upper_bounds)
            else:
                raise ValueError("Invalid initial distribution type. Must be 'repeat', 'gaussian', or 'uniform'.")
        else:
            raise ValueError("Invalid initial state or initial distribution configuration.")

        self.initial_state = initial_state



"""
s = Sampler(log_prob, covmat=scales)
s.set_bounds(lower_bounds, upper_bounds)
s.sample(method='affine', n_steps=1000, n_chains=200)

s.sample(method='affine', n_steps=1000, initial_state=walkers)

s.sample(method='affine', n_steps=1000, n_chains=200, initial_state=mean, initial_distribution='gaussian')

uniform, gaussian, repeat
"""
