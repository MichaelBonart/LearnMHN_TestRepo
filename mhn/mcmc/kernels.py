# author: Y. Linda Hu

import collections
import numpy as np
import scipy.linalg
from typing import Callable


class Kernel:
    """Base class for kernels used in MCMC sampling."""

    def __init__(
        self,
        grad_and_log_likelihood: tuple[Callable[[np.ndarray], tuple[np.ndarray, float]]],
        log_prior: Callable[[np.ndarray], float],
        shape: tuple[int, int],
        rng: np.random.Generator | None = None,
    ):
        self.grad_and_log_likelihood = grad_and_log_likelihood
        self.log_prior = log_prior
        self.rng = rng or np.random.Generator(np.random.PCG64())
        self.shape = shape
        self.size = shape[0]*shape[1]


class smMALAKernel(Kernel):

    Result = collections.namedtuple(
        "smMALAResult",
        [
            "log_likelihood",
            "log_prior",
            "gradient",
            "G",
            "cholesky",
            "mu",
            "det_sqrt",
        ],
    )

    def __init__(
        self,
        step_size: float,
        grad_and_log_likelihood: tuple[Callable[[np.ndarray], tuple[np.ndarray, float]]],
        log_prior: Callable[[np.ndarray], float],
        log_prior_grad: Callable[[np.ndarray], np.ndarray],
        log_prior_hessian: Callable[[np.ndarray], np.ndarray],
        shape: tuple[int, int],
        omhn: bool = True,
        use_cuda: bool = False,
        rng: np.random.Generator | None = None,
    ):

        super().__init__(grad_and_log_likelihood=grad_and_log_likelihood,
                         log_prior=log_prior, shape=shape, rng=rng)

        self.step_size = step_size
        self.log_prior_grad = log_prior_grad
        self.log_prior_hessian = log_prior_hessian
        self.use_cuda = use_cuda
        self.omhn = omhn

    def propose(self, prev_step, prev_step_res):

        # draw random normal number
        z = self.rng.normal(size=prev_step.size)

        # transform with inverse transformed cholesky matrix
        y = np.sqrt(self.step_size) * scipy.linalg.solve_triangular(
            prev_step_res.cholesky.T, z, lower=False
        )

        new_step = prev_step_res.mu + y
        return new_step, self.get_params(new_step)

    def log_accept(self, prev_step, prev_step_res, new_step, new_step_res):
        # p(theta' | D) q(theta | theta') pr(theta') / p(theta|D) q(theta' | theta) pr(theta)

        acceptance_ratio = (
            new_step_res.log_likelihood
            + new_step_res.log_prior
            + self.log_q(
                theta_proposed=prev_step,
                G=new_step_res.G,
                det_sqrt_G=new_step_res.det_sqrt,
                mu=new_step_res.mu,
            )
            - prev_step_res.log_likelihood
            - prev_step_res.log_prior
            - self.log_q(
                theta_proposed=new_step,
                G=prev_step_res.G,
                det_sqrt_G=prev_step_res.det_sqrt,
                mu=prev_step_res.mu,
            )
        )

        return acceptance_ratio

    def one_step(self, prev_step, prev_step_res, return_info=False):

        new_step, new_step_res = self.propose(prev_step, prev_step_res)
        acceptance_ratio = self.log_accept(
            prev_step, prev_step_res, new_step, new_step_res
        )

        if np.log(self.rng.random()) < acceptance_ratio:
            if return_info:
                return (new_step, new_step_res, acceptance_ratio, 1)
            return (new_step, new_step_res)
        else:
            if return_info:
                return (prev_step, prev_step_res, acceptance_ratio, 0)
            return (prev_step, prev_step_res)

    def get_params(self, initial_step):

        # Get gradient, likelihood and G matrix for new theta
        log_likelihood_grad, log_likelihood = self.grad_and_log_likelihood(
            initial_step)
        log_prior = self.log_prior(initial_step)

        log_posterior_grad = log_likelihood_grad + \
            self.log_prior_grad(initial_step)

        fisher = fisher_information_matrix(
            log_theta=initial_step.reshape(self.shape),
            omhn=self.omhn,
            use_cuda=self.use_cuda,
        )
        G = fisher - self.log_prior_hessian(initial_step)
        cholesky = scipy.linalg.cholesky(G, lower=True)
        det_sqrt = np.diag(cholesky).prod()

        # Get mu, the mean of the proposal distribution w.r.t. the new theta
        # this is log_theta + 0.5 * STEP_SIZE * G^-1 * gradient
        y = scipy.linalg.solve_triangular(
            cholesky.T,
            scipy.linalg.solve_triangular(
                cholesky,
                log_posterior_grad.flatten(),
                lower=True,
            ),
            lower=False,
        )
        mu = initial_step.flatten() + 0.5 * self.step_size * y

        return self.Result(
            log_likelihood,
            log_prior,
            log_posterior_grad,
            G,
            cholesky,
            mu,
            det_sqrt,
        )

    def log_q(
        self,
        theta_proposed: np.ndarray,
        G: np.ndarray,
        det_sqrt_G: float,
        mu: np.ndarray,
    ) -> float:
        """Compute the logarithm of the proposal distribution density q(theta_new | theta) for the MMALA algorithm.
        This is according to https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function.

        Args:
            theta_new (np.ndarray): New proposed theta
            G (np.ndarray): Metric tensor w.r.t. the old theta
            det_sqrt_G (float): Square root of the determinant of the metric
            tensor w.r.t. the old theta
            mu (np.ndarray): Mean of the proposal distribution w.r.t. the
            old theta

        Returns:
            float: The logarithm of the proposal distribution q(theta_new | theta) density
        """
        # we can leave out the constant factor (2 * np.pi) ** (n_events**2 )
        # in the denominator, as well as the scaling STEP_SIZE ** (n_events**2)
        # 1/sqrt(det(G^-1)) = sqrt(det(G))
        return -0.5 * (theta_proposed - mu).T @ G @ (
            theta_proposed - mu
        ) / self.step_size + np.log(det_sqrt_G)


class MALAKernel(Kernel):

    Result = collections.namedtuple(
        "MALAResult",
        ["log_likelihood", "log_prior", "mu"],
    )

    def __init__(
        self,
        step_size: float,
        grad_and_log_likelihood: tuple[Callable[[np.ndarray], tuple[np.ndarray, float]]],
        log_prior: Callable[[np.ndarray], float],
        log_prior_grad: Callable[[np.ndarray], np.ndarray],
        shape: tuple[int, int],
        rng: np.random.Generator | None = None,
    ):

        super().__init__(grad_and_log_likelihood=grad_and_log_likelihood,
                         log_prior=log_prior, shape=shape, rng=rng)

        self.step_size = step_size
        self.log_prior_grad = log_prior_grad

    def propose(self, prev_step, prev_step_res):

        z = self.rng.normal(size=self.size)
        new_step = prev_step_res.mu + np.sqrt(self.step_size) * z

        return new_step, self.get_params(new_step)

    def log_accept(self, prev_step, prev_step_res, new_step, new_step_res):
        # p(theta' | D) q(theta | theta') pr(theta') /
        # p(theta|D) q(theta' | theta) pr(theta)

        acceptance_ratio = (
            new_step_res.log_likelihood
            + new_step_res.log_prior
            + self.log_q(
                theta_proposed=prev_step,
                mu=new_step_res.mu,
            )
            - prev_step_res.log_likelihood
            - prev_step_res.log_prior
            - self.log_q(
                theta_proposed=new_step,
                mu=prev_step_res.mu,
            )
        )

        return acceptance_ratio

    def one_step(self, prev_step, prev_step_res, return_info=False):

        new_step, new_step_res = self.propose(prev_step, prev_step_res)
        acceptance_ratio = self.log_accept(
            prev_step, prev_step_res, new_step, new_step_res
        )

        if np.log(self.rng.random()) < acceptance_ratio:
            if return_info:
                return (new_step, new_step_res, acceptance_ratio, 1)
            return (new_step, new_step_res)
        else:
            if return_info:
                return (prev_step, prev_step_res, acceptance_ratio, 0)
            return (prev_step, prev_step_res)

    def get_params(self, initial_step):

        log_likelihood_grad, log_likelihood = self.grad_and_log_likelihood(
            initial_step)
        log_posterior_grad = log_likelihood_grad + \
            self.log_prior_grad(initial_step.reshape(self.shape))

        mu = initial_step.flatten() \
            + 0.5 * self.step_size * log_posterior_grad.flatten()
        return self.Result(
            log_likelihood,
            self.log_prior(initial_step.reshape(self.shape)),
            mu,
        )

    def log_q(
        self,
        theta_proposed: np.ndarray,
        mu: np.ndarray,
    ) -> float:
        """Compute the logarithm of the proposal distribution density
        q(theta_new | theta) for the MMALA algorithm.
        This is according to https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function.

        Args:
            theta_new (np.ndarray): New proposed theta
            G (np.ndarray): Metric tensor w.r.t. the old theta
            det_sqrt_G (float): Square root of the determinant of the
            metric tensor w.r.t. the old theta
            mu (np.ndarray): Mean of the proposal distribution w.r.t.
            the old theta

        Returns:
            float: The logarithm of the proposal distribution 
            q(theta_new | theta) density
        """
        # we can leave out the constant factor (2 * np.pi) ** (n_events**2 )
        # in the denominator, as well as the scaling STEP_SIZE ** (n_events**2)
        # 1/sqrt(det(G^-1)) = sqrt(det(G))
        return -0.5 * np.sum((theta_proposed - mu) ** 2) / self.step_size


class RWMKernel(Kernel):

    Result = collections.namedtuple(
        "RWMResult",
        [
            "log_likelihood",
            "log_prior",
        ],
    )

    def __init__(
        self,
        grad_and_log_likelihood: tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], float]],
        log_prior: Callable[[np.ndarray], float],
        step_size: float,
        shape: tuple[int, int],
        rng: np.random.Generator | None = None,
    ):
        """Initialize the Random Walk Metropolis kernel.

        Parameters
        ----------
        data : np.ndarray
            The data to be used for the sampling.
        sigma : np.ndarray
            The covariance matrix for the proposal distribution.
        prior : dict
            The prior distribution parameters.
        cmhn : bool, optional
            Whether to use the CMHN model. Defaults to False.
        rng : np.random.Generator | None, optional
            The random number generator. If None, a new RNG will be created.
            Defaults to None.
        """

        super().__init__(grad_and_log_likelihood=grad_and_log_likelihood,
                         log_prior=log_prior, shape=shape, rng=rng)

        self.step_size = step_size
        self.sigma = step_size * np.eye(self.size)

    def propose(self, prev_step, prev_step_res):

        # draw random normal number
        new_step = self.rng.normal(
            loc=prev_step,
            scale=self.step_size,
            size=self.size,
        )

        return new_step, self.get_params(new_step)

    def log_accept(self, prev_step, prev_step_res, new_step, new_step_res):
        # p(theta' | D) q(theta | theta') pr(theta') / p(theta|D) q(theta' | theta) pr(theta)

        acceptance_ratio = (
            new_step_res.log_likelihood
            + new_step_res.log_prior
            - prev_step_res.log_likelihood
            - prev_step_res.log_prior
        )

        return acceptance_ratio

    def one_step(self, prev_step, prev_step_res, return_info=False):

        new_step, new_step_res = self.propose(prev_step, prev_step_res)
        acceptance_ratio = self.log_accept(
            prev_step, prev_step_res, new_step, new_step_res
        )

        if np.log(self.rng.random()) < acceptance_ratio:
            if return_info:
                return (new_step, new_step_res, acceptance_ratio, 1)
            return (new_step, new_step_res)
        else:
            if return_info:
                return (prev_step, prev_step_res, acceptance_ratio, 0)
            return (prev_step, prev_step_res)

    def get_params(self, initial_step):

        return self.Result(
            self.grad_and_log_likelihood(initial_step.reshape(self.shape))[1],
            self.log_prior(initial_step.reshape(self.shape))
        )
