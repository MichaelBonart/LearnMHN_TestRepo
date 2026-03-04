"""
Unit tests for the mhn.mcmc module.

This test suite covers:
  - Kernel classes (RWMKernel, MALAKernel, smMALAKernel)
  - MCMC sampler initialization and configuration
  - MCMC sampling with different kernels and parameters
  - Convergence diagnostics (rhat, ESS, acceptance rate)
  - Reproducibility with seeds
  - Pickling/serialization support
  - Custom penalties and priors

Test Classes:
  - TestKernels: Unit tests for all kernel classes (RWM, MALA, smMALA)
  - TestMCMC: Integration tests for MCMC sampler with different kernels and configurations
"""

import unittest
import warnings
import numpy as np
import mhn
from mhn.full_state_space import Likelihood, ModelConstruction
from mhn.training.likelihood_cmhn import gradient_and_score
from mhn.training.state_containers import StateContainer
from mhn import mcmc
from mhn.mcmc.kernels import RWMKernel, MALAKernel, smMALAKernel
import arviz
import tempfile
import os
import scipy.stats

data = np.loadtxt("../demo/LUAD_n12.csv", delimiter=",",
                  skiprows=1, dtype=np.int32)[:100, :3]
data_container = StateContainer(data)
data_size = data.shape[0]
optimizer = mhn.optimizers.oMHNOptimizer().load_data_matrix(data)
lam = optimizer.lambda_from_cv()
model = optimizer.train(lam=lam)
shape = (4, 3)
size = 12
lam = 0.001


def grad_and_log_likelihood(theta):
    grad, lik = mhn.training.likelihood_omhn.gradient_and_score(
        theta.reshape(shape), data_container)
    return data_size * grad, data_size * lik


def log_l2_prior(theta):
    return -lam * np.sum(theta**2)


def log_l2_prior_grad(theta):
    return -2 * lam * theta.reshape(shape)


def log_l2_prior_hessian(theta):
    return -2 * lam * np.eye(size)


def log_l1_prior(theta):
    return -lam * np.sum(np.abs(theta))


def log_l1_prior_grad(theta):
    return -lam * np.sign(theta.reshape(shape))


def log_l1_prior_hessian(theta):
    return np.zeros((size, size))


initial_theta = np.random.normal(loc=0, scale=1 / np.sqrt(2 * lam), size=size)


class TestKernels(unittest.TestCase):
    """Tests for MCMC kernel classes."""

    def test_rwm_kernel_init(self):
        """Test RWMKernel initialization."""
        kernel = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
        )
        self.assertEqual(kernel.step_size, 0.1)
        self.assertEqual(kernel.shape, shape)
        self.assertEqual(kernel.size, size)

    def test_rwm_kernel_get_params(self):
        """Test RWMKernel get_params method."""
        kernel = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
        )

        result = kernel.get_params(initial_theta)

        assert (np.allclose(result.log_likelihood,
                grad_and_log_likelihood(initial_theta)[1]))
        assert (np.allclose(result.log_prior, log_l2_prior(initial_theta)))

    def test_rwm_kernel_propose(self):
        """Test RWMKernel propose method."""
        kernel = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res = kernel.get_params(initial_theta)
        kernel.propose(
            initial_theta, prev_step_res)

    def test_rwm_kernel_log_accept(self):
        """Test RWMKernel log_accept method."""
        kernel = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
        )

        step1 = initial_theta
        res1 = kernel.get_params(step1)
        step2 = initial_theta + 0.1
        res2 = kernel.get_params(step2)

        accept_ratio = kernel.log_accept(step1, res1, step2, res2)

        assert (np.allclose(
            accept_ratio,
            log_l2_prior(step2)
            - log_l2_prior(step1)
            + grad_and_log_likelihood(step2)[1]
            - grad_and_log_likelihood(step1)[1]))

    def test_rwm_kernel_one_step(self):
        """Test RWMKernel one_step method."""
        kernel = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res = kernel.get_params(initial_theta)
        new_step, new_step_res, acceptance_ratio, accepted = kernel.one_step(
            initial_theta, prev_step_res, return_info=True
        )

        # Check output types and shapes
        self.assertEqual(new_step.shape, initial_theta.shape)
        self.assertIsInstance(acceptance_ratio, float)
        self.assertIn(accepted, [0, 1])

    def test_rwm_kernel_seed_reproducibility(self):
        """Test RWMKernel reproducibility with same seed."""
        # First run with seed 42
        kernel1 = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.one_step(initial_theta, prev_step_res1)

        # Second run with same seed 42
        kernel2 = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.one_step(initial_theta, prev_step_res2)

        # Results should be identical
        np.testing.assert_array_equal(new_step1, new_step2)

    def test_rwm_kernel_different_seeds_differ(self):
        """Test that RWMKernel with different seeds produces different results."""
        # First run with seed 42
        kernel1 = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.propose(initial_theta, prev_step_res1)

        # Second run with different seed 99
        kernel2 = RWMKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(99))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.propose(initial_theta, prev_step_res2)

        # Results should be different (with high probability)
        self.assertFalse(np.allclose(new_step1, new_step2))

    def test_mala_kernel_init(self):
        """Test MALAKernel initialization."""
        kernel = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
        )
        self.assertEqual(kernel.step_size, 0.1)
        self.assertEqual(kernel.shape, shape)

    def test_mala_kernel_get_params(self):
        """Test MALAKernel get_params method."""
        kernel = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
        )

        result = kernel.get_params(initial_theta)

        assert (np.allclose(result.log_likelihood,
                grad_and_log_likelihood(initial_theta)[1]))
        assert (np.allclose(result.log_prior, log_l2_prior(initial_theta)))
        self.assertTrue(hasattr(result, "mu"))

    def test_mala_kernel_propose(self):
        """Test MALAKernel propose method."""
        kernel = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res = kernel.get_params(initial_theta)
        kernel.propose(
            initial_theta, prev_step_res)

    def test_mala_kernel_log_accept(self):
        """Test MALAKernel log_accept method."""
        kernel = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
        )

        step1 = initial_theta
        res1 = kernel.get_params(step1)
        step2 = initial_theta + 0.1
        res2 = kernel.get_params(step2)

        accept_ratio = kernel.log_accept(step1, res1, step2, res2)

        assert (np.allclose(
            accept_ratio,
            log_l2_prior(step2)
            - log_l2_prior(step1)
            + grad_and_log_likelihood(step2)[1]
            - grad_and_log_likelihood(step1)[1]
            + scipy.stats.multivariate_normal.logpdf(
                step1,
                mean=step2 + kernel.step_size / 2 *
                (grad_and_log_likelihood(step2)[
                 0] + log_l2_prior_grad(step2)).flatten(),
                cov=kernel.step_size * np.eye(size))
            - scipy.stats.multivariate_normal.logpdf(
                step2,
                mean=step1 + kernel.step_size / 2 *
                (grad_and_log_likelihood(step1)[
                 0] + log_l2_prior_grad(step1)).flatten(),
                cov=kernel.step_size * np.eye(size))
        ))

    def test_mala_kernel_one_step(self):
        """Test MALAKernel one_step method."""
        kernel = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res = kernel.get_params(initial_theta)
        new_step, new_step_res, acceptance_ratio, accepted = kernel.one_step(
            initial_theta, prev_step_res, return_info=True
        )

        # Check output types and shapes
        self.assertEqual(new_step.shape, initial_theta.shape)
        self.assertIsInstance(acceptance_ratio, (float, np.floating))
        self.assertIn(accepted, [0, 1])

    def test_mala_kernel_seed_reproducibility(self):
        """Test MALAKernel reproducibility with same seed."""
        # First run with seed 123
        kernel1 = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(123))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.one_step(initial_theta, prev_step_res1)

        # Second run with same seed 123
        kernel2 = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(123))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.one_step(initial_theta, prev_step_res2)

        # Results should be identical
        np.testing.assert_array_equal(new_step1, new_step2)

    def test_mala_kernel_different_seeds_differ(self):
        """Test that MALAKernel with different seeds produces different results."""
        # First run with seed 123
        kernel1 = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(123))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.propose(initial_theta, prev_step_res1)

        # Second run with different seed 321
        kernel2 = MALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            step_size=0.1,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(321))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.propose(initial_theta, prev_step_res2)

        # Results should be different (with high probability)
        self.assertFalse(np.allclose(new_step1, new_step2))

    def test_smmala_kernel_init(self):
        """Test smMALAKernel initialization."""
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.1,
            shape=shape,
        )
        self.assertEqual(kernel.step_size, 0.1)
        self.assertEqual(kernel.shape, shape)
        self.assertFalse(kernel.use_cuda)

    def test_smmala_kernel_get_params(self):
        """Test smMALAKernel get_params method."""
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.1,
            shape=shape,
        )

        result = kernel.get_params(initial_theta)

        assert (np.allclose(result.log_likelihood,
                grad_and_log_likelihood(initial_theta)[1]))
        assert (np.allclose(result.log_prior, log_l2_prior(initial_theta)))
        assert (np.allclose(result.gradient, grad_and_log_likelihood(
            initial_theta)[0] + log_l2_prior_grad(initial_theta)))
        assert (np.allclose(result.G, -log_l2_prior_hessian(initial_theta) +
                mhn.full_state_space.fisher.fisher(initial_theta.reshape(shape))))
        assert (np.allclose(result.cholesky, np.linalg.cholesky(result.G)))
        assert hasattr(result, 'mu')
        assert (np.allclose(result.det_sqrt, np.sqrt(np.linalg.det(result.G))))

    def test_smmala_kernel_get_params_cuda(self):
        """Test smMALAKernel get_params method with CUDA."""
        if mhn.cuda_available() != mhn.CUDA_AVAILABLE:
            self.skipTest("CUDA not available, skipping CUDA test.")
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.1,
            shape=shape,
        )
        cuda_kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.1,
            shape=shape,
            use_cuda=True
        )

        result = kernel.get_params(initial_theta)
        result_cuda = cuda_kernel.get_params(initial_theta)

        # Check that results are close (accounting for possible floating point differences)
        self.assertTrue(np.allclose(result.log_likelihood,
                        result_cuda.log_likelihood))
        self.assertTrue(np.allclose(result.log_prior, result_cuda.log_prior))
        self.assertTrue(np.allclose(result.gradient, result_cuda.gradient))
        self.assertTrue(np.allclose(result.G, result_cuda.G))
        self.assertTrue(np.allclose(result.cholesky, result_cuda.cholesky))
        self.assertTrue(np.allclose(result.mu, result_cuda.mu))
        self.assertTrue(np.allclose(result.det_sqrt, result_cuda.det_sqrt))

    def test_linalg_error_non_positive_definite(self):
        """Test that smMALAKernel raises LinAlgError for non-positive-definite metric."""

        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l1_prior,
            log_prior_grad=log_l1_prior_grad,
            log_prior_hessian=log_l1_prior_hessian,
            step_size=0.1,
            shape=shape,
        )

        with self.assertRaises(np.linalg.LinAlgError):
            kernel.get_params(initial_theta)

    def test_smmala_kernel_propose(self):
        """Test smMALAKernel propose method."""
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.01,  # Smaller step size for stability
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )

        prev_step_res = kernel.get_params(initial_theta)
        kernel.propose(
            initial_theta, prev_step_res)

    def test_smmala_kernel_log_accept(self):
        """Test smMALAKernel log_accept method."""
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.01,
            shape=shape,
        )

        step1 = initial_theta
        res1 = kernel.get_params(step1)
        step2 = initial_theta + 0.001
        res2 = kernel.get_params(step2)

        accept_ratio = kernel.log_accept(step1, res1, step2, res2)

        assert (np.allclose(
            accept_ratio,
            log_l2_prior(step2)
            - log_l2_prior(step1)
            + grad_and_log_likelihood(step2)[1]
            - grad_and_log_likelihood(step1)[1]
            + scipy.stats.multivariate_normal.logpdf(
                step1,
                mean=step2 + kernel.step_size / 2 * np.linalg.inv(res2.G) @
                (grad_and_log_likelihood(step2)[
                 0] + log_l2_prior_grad(step2)).flatten(),
                cov=kernel.step_size * np.linalg.inv(res2.G))
            - scipy.stats.multivariate_normal.logpdf(
                step2,
                mean=step1 + kernel.step_size / 2 * np.linalg.inv(res1.G) @
                (grad_and_log_likelihood(step1)[
                 0] + log_l2_prior_grad(step1)).flatten(),
                cov=kernel.step_size * np.linalg.inv(res1.G))
        ))

    def test_smmala_kernel_one_step(self):
        """Test smMALAKernel one_step method."""
        kernel = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.001,  # Very small step size for stability
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(42))
        )
        prev_step_res = kernel.get_params(initial_theta)
        new_step, new_step_res, acceptance_ratio, accepted = kernel.one_step(
            initial_theta, prev_step_res, return_info=True
        )

        # Check output types and shapes
        self.assertEqual(new_step.shape, initial_theta.shape)
        self.assertIsInstance(acceptance_ratio, (float, np.floating))
        self.assertIn(accepted, [0, 1])

    def test_smmala_kernel_seed_reproducibility(self):
        """Test smMALAKernel reproducibility with same seed."""
        # First run with seed 456
        kernel1 = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.001,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(456))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.one_step(initial_theta, prev_step_res1)

        # Second run with same seed 456
        kernel2 = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.001,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(456))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.one_step(initial_theta, prev_step_res2)

        # Results should be identical
        np.testing.assert_array_equal(new_step1, new_step2)

    def test_smmala_kernel_different_seeds_differ(self):
        """Test that smMALAKernel with different seeds produces different results."""
        # First run with seed 456
        kernel1 = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.001,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(456))
        )

        prev_step_res1 = kernel1.get_params(initial_theta)
        new_step1, _ = kernel1.propose(initial_theta, prev_step_res1)

        # Second run with different seed 654
        kernel2 = smMALAKernel(
            grad_and_log_likelihood=grad_and_log_likelihood,
            log_prior=log_l2_prior,
            log_prior_grad=log_l2_prior_grad,
            log_prior_hessian=log_l2_prior_hessian,
            step_size=0.001,
            shape=shape,
            rng=np.random.Generator(np.random.PCG64(654))
        )

        prev_step_res2 = kernel2.get_params(initial_theta)
        new_step2, _ = kernel2.propose(initial_theta, prev_step_res2)

        # Results should be different (with high probability)
        self.assertFalse(np.allclose(new_step1, new_step2))


# class TestMCMC(unittest.TestCase):
#     """Tests for MCMC class."""

#     def test_init_from_optimizer_rwm(self):
#         """Test initialization of MCMC from an optimizer with RWM kernel."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             kernel_class=RWMKernel,
#             n_chains=2
#         )
#         self.assertEqual(sampler.n_chains, 2)
#         self.assertEqual(sampler.kernel_class, RWMKernel)

#     def test_init_from_optimizer_mala(self):
#         """Test initialization of MCMC from an optimizer with MALA kernel."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             kernel_class=MALAKernel,
#             n_chains=2
#         )
#         self.assertEqual(sampler.kernel_class, MALAKernel)

#     def test_init_from_model_and_data(self):
#         """Test initialization of MCMC from a model and data."""
#         sampler = mcmc.mcmc.MCMC(
#             mhn_model=self.model,
#             data=self.data,
#             penalty=self.optimizer.penalty,
#             kernel_class=RWMKernel,
#             n_chains=2
#         )
#         self.assertIsNotNone(sampler.optimizer)
#         self.assertEqual(sampler.n_chains, 2)

#     def test_init_invalid_args(self):
#         """Test that proper errors are raised for invalid arguments."""
#         # Missing data when providing model
#         with self.assertRaises(ValueError):
#             mcmc.mcmc.MCMC(mhn_model=self.model)

#         # Providing both optimizer and model
#         with self.assertRaises(ValueError):
#             mcmc.mcmc.MCMC(
#                 optimizer=self.optimizer,
#                 mhn_model=self.model,
#                 data=self.data
#             )

#         with self.assertRaises(ValueError):
#             mcmc.mcmc.MCMC(
#                 optimizer=self.optimizer,
#                 kernel_class=smMALAKernel,
#                 n_chains=2
#             )

#     def test_step_size_options(self):
#         """Test different step size specifications."""
#         # Fixed step size
#         sampler1 = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.05,
#             n_chains=3
#         )
#         self.assertEqual(sampler1.step_size, 0.05)

#         # Array of step sizes
#         step_sizes = np.array([0.01, 0.05, 0.1])
#         sampler2 = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=step_sizes,
#             n_chains=3
#         )
#         np.testing.assert_array_equal(sampler2.step_size, step_sizes)

#         # Auto step size
#         sampler3 = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size="auto",
#             n_chains=2
#         )
#         self.assertEqual(sampler3.step_size, "auto")

#     def test_thin_parameter(self):
#         """Test thin parameter."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             thin=10,
#             n_chains=2
#         )
#         self.assertEqual(sampler.thin, 10)

#     def test_seed_reproducibility(self):
#         """Test that seed parameter ensures reproducibility."""
#         sampler1 = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             seed=42,
#             n_chains=2,
#             thin=1
#         )
#         sampler1.run(stopping_crit=None, max_steps=10, verbose=False)
#         samples1 = sampler1.log_thetas.copy()

#         sampler2 = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             seed=42,
#             n_chains=2,
#             thin=1
#         )
#         sampler2.run(stopping_crit=None, max_steps=10, verbose=False)
#         samples2 = sampler2.log_thetas.copy()

#         np.testing.assert_array_equal(samples1, samples2)

#     def test_run_rwm_basic(self):
#         """Test basic MCMC run with RWM kernel."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             kernel_class=RWMKernel,
#             step_size=0.05,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )
#         result = sampler.run(stopping_crit=None, max_steps=10, verbose=False)

#         self.assertEqual(result.shape[0], 2)  # n_chains
#         self.assertEqual(result.shape[2], sampler.size)  # parameters

#     def test_run_mala_basic(self):
#         """Test basic MCMC run with MALA kernel."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             kernel_class=MALAKernel,
#             step_size=0.05,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )
#         result = sampler.run(stopping_crit=None, max_steps=10, verbose=False)

#         self.assertEqual(result.shape[0], 2)  # n_chains
#         self.assertGreater(result.shape[1], 0)  # some samples

#     def test_run_smmala_basic(self):
#         """Test basic MCMC run with smMALA kernel."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             kernel_class=smMALAKernel,
#             step_size=0.001,  # Smaller step size for smMALA
#             n_chains=2,
#             thin=1,
#             seed=42
#         )
#         try:
#             result = sampler.run(stopping_crit=None,
#                                  max_steps=10, verbose=False)

#             self.assertEqual(result.shape[0], 2)  # n_chains
#             self.assertGreater(result.shape[1], 0)  # some samples
#         except np.linalg.LinAlgError:
#             # smMALA may fail with some models if metric tensor not positive definite
#             pass

#     def test_run_with_max_steps(self):
#         """Test run with max_steps limit."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=2,
#             thin=2,
#             seed=42
#         )
#         sampler.run(stopping_crit=None, max_steps=20, verbose=False)

#         # Should have at most 20/2 = 10 samples per chain
#         self.assertLessEqual(sampler.log_thetas.shape[1], 11)

#     def test_rhat_computation(self):
#         """Test rhat computation."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=3,
#             thin=1,
#             seed=42
#         )
#         sampler.run(stopping_crit=None, max_steps=50, verbose=False)

#         rhat_values = sampler.rhat()

#         # rhat should be array with one value per parameter
#         self.assertEqual(len(rhat_values), sampler.size)
#         # All rhat values should be positive
#         self.assertTrue(np.all(rhat_values > 0))

#     def test_rhat_with_burn_in_int(self):
#         """Test rhat with integer burn-in."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )
#         sampler.run(stopping_crit=None, max_steps=50, verbose=False)

#         rhat_values = sampler.rhat(burn_in=10)

#         self.assertEqual(len(rhat_values), sampler.size)
#         self.assertTrue(np.all(rhat_values > 0))

#     def test_rhat_with_burn_in_float(self):
#         """Test rhat with fractional burn-in."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )
#         sampler.run(stopping_crit=None, max_steps=50, verbose=False)

#         rhat_values = sampler.rhat(burn_in=0.2)

#         self.assertEqual(len(rhat_values), sampler.size)

#     def test_ess_computation(self):
#         """Test effective sample size computation."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )
#         sampler.run(stopping_crit=None, max_steps=100, verbose=False)

#         ess_values = sampler.ess()

#         # ESS should be array with one value per parameter
#         self.assertEqual(len(ess_values), sampler.size)
#         # All ESS values should be positive
#         self.assertTrue(np.all(ess_values > 0))
#         # ESS should be less than total samples
#         total_samples = sampler.log_thetas.shape[0] * \
#             sampler.log_thetas.shape[1]
#         self.assertTrue(np.all(ess_values <= total_samples))

#     def test_ess_with_burn_in(self):
#         """Test ESS with burn-in."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )
#         sampler.run(stopping_crit=None, max_steps=100, verbose=False)

#         ess_values = sampler.ess(burn_in=0.3)

#         self.assertEqual(len(ess_values), sampler.size)
#         self.assertTrue(np.all(ess_values > 0))

#     def test_acceptance_rate_all_chains(self):
#         """Test acceptance rate computation for all chains."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=3,
#             thin=1,
#             seed=42
#         )
#         sampler.run(stopping_crit=None, max_steps=50, verbose=False)

#         acceptance_rates = sampler.acceptance()

#         # Should have one rate per chain
#         self.assertEqual(len(acceptance_rates), 3)
#         # All rates should be between 0 and 1
#         self.assertTrue(np.all(acceptance_rates >= 0))
#         self.assertTrue(np.all(acceptance_rates <= 1))

#     def test_acceptance_rate_single_chain(self):
#         """Test acceptance rate for a single chain."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=3,
#             thin=1,
#             seed=42
#         )
#         sampler.run(stopping_crit=None, max_steps=50, verbose=False)

#         acceptance_rate = sampler.acceptance(chain_id=0)

#         # Should be a single float
#         self.assertIsInstance(acceptance_rate, (float, np.floating))
#         # Should be between 0 and 1
#         self.assertTrue(0 <= acceptance_rate <= 1)

#     def test_acceptance_rate_with_burn_in(self):
#         """Test acceptance rate with burn-in."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )
#         sampler.run(stopping_crit=None, max_steps=50, verbose=False)

#         acceptance_rates = sampler.acceptance(burn_in=0.2)

#         self.assertEqual(len(acceptance_rates), 2)
#         self.assertTrue(np.all(acceptance_rates >= 0))
#         self.assertTrue(np.all(acceptance_rates <= 1))

#     # def test_convergence_rhat_criterion(self):
#     #     """Test run with rhat convergence criterion."""
#     #     sampler = mcmc.mcmc.MCMC(
#     #         optimizer=self.optimizer,
#     #         step_size=0.01,
#     #         n_chains=2,
#     #         thin=10,
#     #         seed=42
#     #     )
#     #     sampler.run(
#     #         stopping_crit="r_hat",
#     #         max_steps=200,
#     #         check_interval=50,
#     #         verbose=False
#     #     )

#     #     # Should have run some iterations
#     #     self.assertGreater(sampler.log_thetas.shape[1], 0)

#     # def test_convergence_ess_criterion(self):
#     #     """Test run with ESS convergence criterion."""
#     #     sampler = mcmc.mcmc.MCMC(
#     #         optimizer=self.optimizer,
#     #         step_size=0.01,
#     #         n_chains=2,
#     #         thin=10,
#     #         seed=42
#     #     )
#     #     sampler.run(
#     #         stopping_crit="ESS",
#     #         max_steps=500,
#     #         check_interval=50,
#     #         verbose=False
#     #     )

#     #     # Should have run some iterations
#     #     self.assertGreater(sampler.log_thetas.shape[1], 0)

#     def test_smmala_kernel_in_mcmc(self):
#         """Test smMALA kernel within MCMC framework."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             kernel_class=smMALAKernel,
#             step_size=0.001,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )

#         try:
#             sampler.run(stopping_crit=None, max_steps=10, verbose=False)

#             # Check sampler state
#             self.assertEqual(sampler.kernel_class, smMALAKernel)
#             self.assertGreater(sampler.log_thetas.shape[1], 0)
#             self.assertEqual(sampler.log_thetas.shape[0], 2)  # n_chains
#         except np.linalg.LinAlgError:
#             # Expected if Fisher matrix computation fails
#             pass

#     def test_smmala_with_hessian_penalty(self):
#         """Test smMALA with Hessian of penalty."""
#         def custom_penalty(theta):
#             return np.sum(theta**2)

#         def custom_penalty_grad(theta):
#             return 2 * theta

#         def custom_penalty_hessian(theta):
#             return 2 * np.eye(theta.size)

#         sampler = mcmc.mcmc.MCMC(
#             mhn_model=self.model,
#             data=self.data,
#             penalty=(custom_penalty, custom_penalty_grad,
#                      custom_penalty_hessian),
#             kernel_class=smMALAKernel,
#             n_chains=2
#         )

#         # Set initial step manually
#         sampler.initial_step = self.optimizer.result.log_theta.flatten().reshape(
#             2, 1, -1
#         )

#         try:
#             sampler.run(stopping_crit=None, max_steps=10, verbose=False)

#             self.assertGreater(sampler.log_thetas.shape[1], 0)
#         except np.linalg.LinAlgError:
#             # Expected if metric tensor not positive definite
#             pass

#     def test_pickle_support(self):
#         """Test that MCMC sampler can be pickled and unpickled."""
#         sampler1 = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )
#         sampler1.run(stopping_crit=None, max_steps=20, verbose=False)

#         # Pickle and unpickle
#         import pickle
#         with tempfile.NamedTemporaryFile(delete=False) as f:
#             pickle.dump(sampler1, f)
#             temp_file = f.name

#         try:
#             with open(temp_file, 'rb') as f:
#                 sampler2 = pickle.load(f)

#             # Check that loaded sampler has the same data
#             np.testing.assert_array_equal(
#                 sampler1.log_thetas,
#                 sampler2.log_thetas
#             )
#             self.assertEqual(sampler1.n_chains, sampler2.n_chains)
#         finally:
#             os.unlink(temp_file)

#     def test_custom_penalty(self):
#         """Test MCMC with custom penalty."""
#         def custom_penalty(theta):
#             return np.sum(np.abs(theta))

#         def custom_penalty_grad(theta):
#             return np.sign(theta)

#         sampler = mcmc.mcmc.MCMC(
#             mhn_model=self.model,
#             data=self.data,
#             penalty=(custom_penalty, custom_penalty_grad),
#             kernel_class=MALAKernel,
#             n_chains=2
#         )

#         # Need to set initial step manually for custom penalty
#         sampler.initial_step = self.optimizer.result.log_theta.flatten().reshape(
#             2, 1, -1
#         )
#         sampler.run(stopping_crit=None, max_steps=10, verbose=False)

#         self.assertGreater(sampler.log_thetas.shape[1], 0)

#     def test_initial_step_setting(self):
#         """Test setting initial step values."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=2
#         )

#         # Set custom initial step
#         initial_step = np.random.randn(2, 1, sampler.size)
#         sampler.initial_step = initial_step

#         sampler.run(stopping_crit=None, max_steps=10, verbose=False)

#         # First step should come from initial_step
#         np.testing.assert_array_almost_equal(
#             sampler.log_thetas[:, 0, :],
#             initial_step[:, 0, :]
#         )

#     def test_multiple_runs(self):
#         """Test that multiple consecutive runs accumulate samples."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )

#         sampler.run(stopping_crit=None, max_steps=10, verbose=False)
#         n_samples_1 = sampler.log_thetas.shape[1]

#         sampler.run(stopping_crit=None, max_steps=10, verbose=False)
#         n_samples_2 = sampler.log_thetas.shape[1]

#         # Second run should add more samples
#         self.assertGreater(n_samples_2, n_samples_1)

#     def test_single_chain(self):
#         """Test with single chain."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             n_chains=1,
#             step_size=0.01,
#             thin=1,
#             seed=42
#         )
#         sampler.run(stopping_crit=None, max_steps=10, verbose=False)

#         self.assertEqual(sampler.log_thetas.shape[0], 1)

#     def test_large_thin_factor(self):
#         """Test with large thin factor."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             n_chains=2,
#             step_size=0.01,
#             thin=100,
#             seed=42
#         )
#         sampler.run(stopping_crit=None, max_steps=100, verbose=False)

#         # With thin=100 and max_steps=100, we should get ~1 sample per chain
#         self.assertLessEqual(sampler.log_thetas.shape[1], 2)

#     def test_zero_max_steps(self):
#         """Test with max_steps=0."""
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             step_size=0.01,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )
#         result = sampler.run(stopping_crit=None, max_steps=0, verbose=False)

#         # Should return empty or initial samples
#         self.assertEqual(result.shape[0], 2)

#     def test_smmala_kernel_with_different_step_sizes(self):
#         """Test smMALA with different step sizes per chain."""
#         step_sizes = np.array([0.0001, 0.001])
#         sampler = mcmc.mcmc.MCMC(
#             optimizer=self.optimizer,
#             kernel_class=smMALAKernel,
#             step_size=step_sizes,
#             n_chains=2,
#             thin=1,
#             seed=42
#         )

#         try:
#             sampler.run(stopping_crit=None, max_steps=10, verbose=False)

#             self.assertEqual(sampler.log_thetas.shape[0], 2)
#             self.assertGreater(sampler.log_thetas.shape[1], 0)
#         except np.linalg.LinAlgError:
#             # Expected if metric tensor not positive definite
#             pass


if __name__ == "__main__":
    unittest.main()
