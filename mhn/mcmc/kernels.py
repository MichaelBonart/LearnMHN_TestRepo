import collections
import mhn
from mhn.training.state_containers import StateContainer
from mhn.training.likelihood_cmhn import gradient_and_score as cmhn_gradient_and_score
from mhn.training.likelihood_omhn import gradient_and_score as omhn_gradient_and_score
import numpy as np
import scipy.linalg


class Kernel:
    """Base class for kernels used in MALA sampling."""

    def __init__(
        self,
        data: StateContainer,
        prior: dict,
        omhn=True,
        rng: np.random.Generator | None = None,
    ):
        self.data = data
        self.prior = prior
        self.omhn = omhn
        self.rng = rng or np.random.Generator(np.random.PCG64())


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
        data: StateContainer,
        step_size: float,
        prior: dict,
        omhn=True,
        rng: np.random.Generator | None = None,
    ):

        super().__init__(data=data, prior=prior, omhn=omhn, rng=rng)

        self.step_size = step_size
        _n_events = data.get_data_shape()[1]
        self._theta_x = _n_events + 1 if omhn else _n_events
        self._theta_y = _n_events

        self._unscaled_gradient_likelihood = \
            self._get_gradient_and_score()

        self.log_prior, self.log_prior_grad, self.log_prior_hessian = (
            get_log_prior_grad_hessian(
                n=_n_events,
                **prior,
                omhn=omhn,
            )
        )

        # For more than 11 events, the CUDA implementation is faster.
        self._use_cuda = _n_events >= 13 and (
            mhn.cuda_available() == mhn.CUDA_AVAILABLE
        )

    @staticmethod
    def _get_log_likelihood(
        data_container: mhn.training.state_containers.StateContainer,
        cmhn=False,
    ):

        n_patients = data_container.get_data_shape()[0]
        n_events = data_container.get_data_shape()[1]

        if cmhn:

            def log_likelihood(log_theta: np.ndarray) -> float:
                _, log_likelihood = (
                    mhn.training.likelihood_cmhn.gradient_and_score(
                        log_theta.reshape(n_events, n_events),
                        data_container,
                    )
                )
                return n_patients * log_likelihood

        else:

            def log_likelihood(log_theta: np.ndarray) -> float:
                _, log_likelihood = (
                    mhn.training.likelihood_omhn.gradient_and_score(
                        log_theta.reshape(n_events + 1, n_events),
                        data_container,
                    )
                )
                return n_patients * log_likelihood

        return log_likelihood

    def _get_gradient_and_score(self):

        n_patients = self._data_container.get_data_shape()[0]

        if self.omhn:

            def gradient_and_score(
                log_theta: np.ndarray,
            ) -> tuple[np.array, float]:
                gradient, log_likelihood = (
                    mhn.training.likelihood_omhn.gradient_and_score(
                        log_theta.reshape(self._theta_x, self._theta_y),
                        self._data_container,
                    )
                )
                return (
                    1 / self.temperature * (n_patients * gradient),
                    1 / self.temperature * (n_patients * log_likelihood),
                )

        else:

            def gradient_and_score(
                log_theta: np.ndarray,
            ) -> tuple[np.array, float]:
                gradient, log_likelihood = (
                    mhn.training.likelihood_cmhn.gradient_and_score(
                        log_theta.reshape(self._theta_x, self._theta_y),
                        self._data_container,
                    )
                )
                return (
                    1 / self.temperature * (n_patients * gradient),
                    1 / self.temperature * (n_patients * log_likelihood),
                )

        return gradient_and_score

    def propose(self, prev_step, prev_step_res):

        # draw random normal number
        z = self.rng.normal(size=self._theta_x * self._theta_y)

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
        gradient, log_likelihood = self.gradient_and_score(initial_step)
        log_prior = self.log_prior(initial_step)

        grad_posterior = gradient + self.log_prior_grad(initial_step)

        if not self.empirical_fisher:
            fisher = fisher_information_matrix(
                log_theta=initial_step.reshape(self._theta_x, self._theta_y),
                omhn=not self.cmhn,
                use_cuda=self._use_cuda,
            )
        else:
            fisher = empirical_fisher_information_matrix(
                log_theta=initial_step.reshape(self._theta_x, self._theta_y),
                omhn=not self.cmhn,
                use_cuda=self._use_cuda,
                num_samples=self.empirical_fisher
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
                grad_posterior.flatten(),
                lower=True,
            ),
            lower=False,
        )
        mu = initial_step.flatten() + 0.5 * self.step_size * y

        return self.Result(
            log_likelihood,
            log_prior,
            grad_posterior,
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
        ["log_likelihood", "log_prior", "gradient", "mu"],
    )

    def __init__(
        self,
        data: np.ndarray,
        step_size: float,
        prior: dict,
        omhn=True,
        rng: np.random.Generator | None = None,
        temperature: float = 1.0,
    ):

        super().__init__(data=data, prior=prior, omhn=omhn, rng=rng)

        self.step_size = step_size
        self._temperature = temperature
        _n_events = data.shape[1]
        self._theta_x = data.shape[1] if omhn else data.shape[1] + 1
        self._theta_y = data.shape[1]

        self._data_container = mhn.training.state_containers.StateContainer(
            data
        )

        self.gradient_and_score = self._get_gradient_and_score()

        self.log_prior, self.log_prior_grad, self.log_prior_hessian = \
            get_log_prior_grad_hessian(
                n=_n_events,
                **prior,
                omhn=omhn,
            )

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value == self._temperature:
            return
        self._temperature = value
        self.gradient_and_score = self._get_gradient_and_score()

    @staticmethod
    def _get_log_likelihood(
        data_container: mhn.training.state_containers.StateContainer,
        cmhn=False,
    ):

        n_patients = data_container.get_data_shape()[0]
        n_events = data_container.get_data_shape()[1]

        if cmhn:

            def log_likelihood(log_theta: np.ndarray) -> float:
                _, log_likelihood = (
                    mhn.training.likelihood_cmhn.gradient_and_score(
                        log_theta.reshape(n_events, n_events),
                        data_container,
                    )
                )
                return n_patients * log_likelihood

        else:

            def log_likelihood(log_theta: np.ndarray) -> float:
                _, log_likelihood = (
                    mhn.training.likelihood_omhn.gradient_and_score(
                        log_theta.reshape(n_events + 1, n_events),
                        data_container,
                    )
                )
                return n_patients * log_likelihood

        return log_likelihood

    def _get_gradient_and_score(self):

        n_patients = self._data_container.get_data_shape()[0]

        if self.omhn:

            def gradient_and_score(
                log_theta: np.ndarray,
            ) -> tuple[np.array, float]:
                gradient, log_likelihood = (
                    mhn.training.likelihood_omhn.gradient_and_score(
                        log_theta.reshape(self._theta_x, self._theta_y),
                        self._data_container,
                    )
                )
                return (
                    1 / self.temperature * (n_patients * gradient),
                    1 / self.temperature * (n_patients * log_likelihood),
                )

        else:

            def gradient_and_score(
                log_theta: np.ndarray,
            ) -> tuple[np.array, float]:
                gradient, log_likelihood = (
                    mhn.training.likelihood_cmhn.gradient_and_score(
                        log_theta.reshape(self._theta_x, self._theta_y),
                        self._data_container,
                    )
                )
                return (
                    1 / self.temperature * (n_patients * gradient),
                    1 / self.temperature * (n_patients * log_likelihood),
                )

        return gradient_and_score

    def propose(self, prev_step, prev_step_res):

        # draw random normal number
        z = self.rng.normal(size=self._theta_x * self._theta_y)

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

        # Get gradient, likelihood and G matrix for new theta
        gradient, log_likelihood = self.gradient_and_score(initial_step)
        grad_posterior = gradient + self.log_prior_grad(initial_step)
        log_prior = self.log_prior(initial_step)

        mu = initial_step.flatten() \
            + 0.5 * self.step_size * grad_posterior.flatten()

        return self.Result(
            log_likelihood,
            log_prior,
            gradient,
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
        data: np.ndarray,
        sigma: np.ndarray,
        prior: dict,
        cmhn=False,
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

        super().__init__(data=data, prior=prior, cmhn=cmhn, rng=rng)

        _n_events = data.shape[1]
        self._theta_x = data.shape[1] if cmhn else data.shape[1] + 1
        self._theta_y = data.shape[1]
        self.sigma = (
            sigma
            if sigma is not None
            else np.eye(self._theta_x * self._theta_y)
        )

        data_container = mhn.training.state_containers.StateContainer(data)

        if cmhn:

            def score(
                log_theta: np.ndarray,
            ) -> tuple[np.array, float]:
                gradient, log_likelihood = (
                    mhn.training.likelihood_cmhn.gradient_and_score(
                        log_theta.reshape(self._theta_x, self._theta_y),
                        data_container,
                    )
                )
                return len(data) * log_likelihood

        else:

            def score(
                log_theta: np.ndarray,
            ) -> tuple[np.array, float]:
                gradient, log_likelihood = (
                    mhn.training.likelihood_omhn.gradient_and_score(
                        log_theta.reshape(self._theta_x, self._theta_y),
                        data_container,
                    )
                )
                return len(data) * log_likelihood

        self.score = score

        self.log_prior, _, _ = get_log_prior_grad_hessian(
            n=_n_events, **prior, omhn=not cmhn
        )

    def propose(self, prev_step, prev_step_res):

        # draw random normal number
        new_step = self.rng.multivariate_normal(
            mean=prev_step,
            cov=self.sigma,
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

        # Get gradient, likelihood and G matrix for new theta
        log_likelihood = self.score(initial_step)
        log_prior = self.log_prior(initial_step)

        return self.Result(
            log_likelihood,
            log_prior,
        )


class AdaptiveMetropolisKernel(MetropolisKernel):

    def __init__(
        self,
        data: np.ndarray,
        prior: dict,
        cov_mat: np.ndarray,
        cmhn=False,
        beta=0.05,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            data=data,
            sigma=np.eye(data.shape[1]),
            prior=prior,
            cmhn=cmhn,
            rng=rng,
        )
        self.beta = beta
        self._dim = self._theta_x * self._theta_y
        self.cov_mat = (
            cov_mat
            if cov_mat is not None
            else 0.01 / self._dim * np.eye(self._dim)
        )

    def propose(self, prev_step, prev_step_res):
        """
        Propose a new step based on the previous step and the covariance matrix.
        This adaptive Metropolis proposition follows Roberts and Rosenthal (2006/2008)
        """
        if self.beta == 1:
            new_step = self.beta * self.rng.multivariate_normal(
                mean=prev_step,
                cov=(0.01 / self._dim) * np.eye(self._dim),
            )
        else:
            new_step = (1 - self.beta) * self.rng.multivariate_normal(
                mean=prev_step,
                cov=(2.38**2 / self._dim) * self.cov_mat,
            ) + self.beta * self.rng.multivariate_normal(
                mean=prev_step,
                cov=(0.01 / self._dim) * np.eye(self._dim),
            )
        new_step_res = self.get_params(new_step)
        return new_step, new_step_res
