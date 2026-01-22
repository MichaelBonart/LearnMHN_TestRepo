import pints
from ..optimizers import Optimizer, Penalty, oMHNOptimizer, cMHNOptimizer
from ..model import cMHN, oMHN
from .posterior import MHNPosterior
import numpy as np
from numpy.typing import ArrayLike
from typing import Literal, Callable
import itertools as it


class MCMC(pints.MCMCController):

    def __init__(
        self,
        optimizer: Optimizer = None,
        mhn: cMHN | oMHN = None,
        data: ArrayLike = None,
        penalty: Penalty | tuple[
            Callable[[np.ndarray], float],
            Callable[[np.ndarray], np.ndarray],
        ] = None,
        chains: int = 10,
        init: ArrayLike = None,
        epsilon: float | None | Literal["auto"] = "auto",
    ):
        """
        Initializes the MCMC controller with the given log-posterior and initial parameters.

        :param log_posterior: An instance of pints.LogPosterior
        :param initial_parameters: A list of initial parameter values
        :param method: The MCMC method to use (default is 'AdaptiveCovarianceMCMC')
        :param kwargs: Additional keyword arguments for the MCMC controller
        """

        if optimizer is None:
            if mhn is None or data is None or penalty is None:
                raise ValueError(
                    "Either optimizer or (mhn, data, penalty) must be provided."
                )
            
            optimizer = oMHNOptimizer() if isinstance(mhn, oMHN) else cMHNOptimizer()
            optimizer.load_data_matrix(data)
            optimizer._result = mhn
            optimizer.set_penalty(penalty)

        elif mhn is not None or data is not None or penalty is not None:
            raise ValueError(
                "Provide either optimizer or (mhn, data, penalty)"
            )

        self.posterior = MHNPosterior(optimizer)
        super().__init__(
            self.posterior,
            method=pints.MALAMCMC,
            chains=chains,
            x0=init
            or self.posterior.prior.draw(chains))
        if epsilon == "auto":
            epsilon = self.tune_epsilon()

        if isinstance(epsilon, (float, int)):
            for sampler in self.samplers():
                sampler.set_epsilon([epsilon] * self.posterior.n_parameters())

    def tune_epsilon(
        self,
        target_accept=0.574,
        n_iter=500,
        max_trials=10,
        tol=0.02,
        verbose=True
    ):
        """
        Tune the step-size epsilon for MALAMCMC to reach target acceptance rate.

        Parameters
        ----------
        target_accept : float
            Desired acceptance rate (default 0.574 for MALA)
        n_iter : int
            Number of iterations per trial run
        max_trials : int
            Maximum number of adjustment steps
        tol : float
            Tolerance for acceptance rate
        verbose : bool
            Print progress info

        Returns
        -------
        epsilon : float
            Tuned step-size
        """

        n_parallel = 10

        epsilons = 10 ** np.linspace(-1, -5, num=n_parallel)

        for trial in range(max_trials):
            if verbose:
                print(f"Trial {trial+1}: epsilons={epsilons}")

            temp_controller = pints.MCMCController(
                self.posterior,
                method=pints.MALAMCMC,
                chains=n_parallel,
                x0=[sampler._x0 for _, sampler in zip(
                    range(n_parallel), it.cycle(self.samplers()))],
            )
            temp_controller.set_log_to_screen(False)
            for sampler, epsilon in zip(temp_controller.samplers(), epsilons):
                sampler.set_epsilon([epsilon] * self.posterior.n_parameters())
            temp_controller.set_max_iterations(n_iter)
            temp_controller.run()

            acceptance_rates = np.array(
                [sampler.acceptance_rate() for sampler in temp_controller.samplers()])

            argbest = np.argmin(np.abs(acceptance_rates - target_accept))
            if np.abs(acceptance_rates[argbest] - target_accept) < tol:
                return epsilons[argbest]
            elif argbest == 0:
                epsilons = np.linspace(
                    epsilons[0] / 10, epsilons[1], n_parallel)
            elif argbest == n_parallel - 1:
                epsilons = np.linspace(
                    epsilons[-2], epsilons[-1] * 10, n_parallel)
            else:
                center = argbest
                epsilons = np.linspace(
                    epsilons[center - 1], epsilons[center + 1], n_parallel)
