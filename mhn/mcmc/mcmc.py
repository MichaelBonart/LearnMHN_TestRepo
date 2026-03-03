"""
This submodule contains the MCMC class that runs Markov Chain Monte
Carlo sampling for MHNs.

"""


# author: Y. Linda Hu

from ..optimizers import Optimizer, oMHNOptimizer, cMHNOptimizer, Penalty
from ..model import oMHN, cMHN
from ..training.likelihood_cmhn \
    import gradient_and_score as cmhn_grad_and_log_likelihood
from ..training.likelihood_omhn \
    import gradient_and_score as omhn_grad_and_log_likelihood
from ..training.state_containers import StateContainer
from typing import Callable, Literal
import numpy as np
import multiprocessing as mp
from .kernels import Kernel, smMALAKernel, RWMKernel, MALAKernel
from ..training import penalties_cmhn, penalties_omhn


class MCMC:
    """Markov chain Monte Carlo sampler for MHN.

    The simplest way to create an MCMC sampler is to provide a trained
    ``Optimizer``. This already includes the dataset, the regularization
    strength lambda, the penalty and its gradient.

    This is enough to create an RWM or MALA sampler. For an smMALA
    sampler, the Hessian of the penalty is also needed, which is not
    stored in the optimizer. In this case, please provide the penalty
    and its derivatives directly via ``penalty`` or ``log_prior``.

    Alternatively, you can create an MCMC sampler from a trained MHN
    model, the data used to train it, and the penalty used for training.
    Depending on the Kernel (RWM, MALA, or smMALA), you have to provide
    the penalty, its gradient, and its Hessian,

    - by providing ``penalty`` either as a :class:`Penalty`
      (only for RWM/MALA) or as a tuple of callables of length 1, 2,
      or 3, or
    - by providing ``log_prior`` as a tuple of callables of length 1,
      2, or 3.
    The difference between ``penalty`` and ``log_prior`` is that the
    former is unscaled by :math:`\\lambda` and positive (such as it is
    stored in an ``Optimizer`` object), while the latter is scaled by
    lambda and negative.

    When using a custom penalty or prior, initial chain values must be
    set manually by setting ``Sampler.initial_step`` to an array of
    shape (n_chains, 1, m) with m the number of parameters.

    Args:
        optimizer (Optimizer, optional): Trained Optimizer.
        mhn_model (oMHN | cMHN, optional): MHN model.
        data (np.ndarray | StateContainer, optional): Data used to train
            the MHN model.
        penalty (Penalty | tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]], optional):
            Penalty used during training. If not Penalty, ``penalty[0]``
            gives the penalty (positive and unscaled by
            :math:`\\lambda`), ``penalty[1]`` its gradient and
            ``penalty[2]`` its Hessian.
        log_prior: Log-prior used during training. ``log_prior[0]``
            gives the log_prior (negative and scaled by 
            :math:`\\lambda`), ``penalty[1]`` its gradient and
            ``penalty[2]`` its Hessian.
        n_chains (int, optional): Number of parallel chains to run.
            Defaults to ``10``.
        step_size: (Literal["auto"] | int | np.ndarray | ) Stepsize of the Kernel. See the documentation of the
            kernels (:class:`RWMKernel<mhn.mcmc.kernels.RWMKernel>`,
            :class:`MALAKernel<mhn.mcmc.kernels.MALAKernel>`,
            :class:`smMALAKernel<mhn.mcmc.kernels.smMALAKernel>`) for
            more details. If ``"auto"``, the stepsize is tuned
            automatically before the first run using
            :func:`tune_stepsize`. If array, every chain uses a
            different step size. Defaults to ``"auto"``.
        kernel_class (Kernel, optional): Kernel class to use for MCMC
            sampling. Must be one of ``RWMKernel``, ``MALAKernel``,
            ```smMALAKernel``. Defaults to ``MALAKernel``.
        thin (int, optional): Thinning factor for MCMC sampling.
            Defaults to ``100``.
        seed (int, optional): Random seed for reproducibility. Defaults
            to ``None``.
    """

    import arviz

    _kernel_args = {
        MALAKernel: ["log_prior_grad"],
        smMALAKernel: ["log_prior_grad", "log_prior_hessian"],
        RWMKernel: []
    }

    _penalty_dict = {
        cMHNOptimizer: {
            Penalty.L1: (penalties_cmhn.l1, penalties_cmhn.l1_),
            Penalty.L2: (penalties_cmhn.l2, penalties_cmhn.l2_),
            Penalty.SYM_SPARSE: (
                penalties_cmhn.sym_sparse,
                penalties_cmhn.sym_sparse_deriv,
            ),
        },
        oMHNOptimizer: {
            Penalty.L1: (penalties_omhn.l1, penalties_omhn.l1_),
            Penalty.L2: (penalties_omhn.l2, penalties_omhn.l2_),
            Penalty.SYM_SPARSE: (
                penalties_omhn.sym_sparse,
                penalties_omhn.sym_sparse_deriv,
            ),
        },
    }

    def __init__(
            self, *,
            optimizer: Optimizer = None,
            mhn_model: oMHN | cMHN | None = None,
            data: np.ndarray | StateContainer | None = None,
            penalty: Penalty | tuple[Callable] | None = None,
            log_prior: tuple[Callable] | None = None,
            n_chains: int = 10,
            step_size: Literal["auto"] | float | np.ndarray = "auto",
            kernel_class: Kernel = MALAKernel,
            thin: int = 100,
            seed: int | None = None,) -> None:

        if optimizer is None:
            if mhn_model is None or data is None:
                raise ValueError(
                    "Either optimizer or (mhn_model, data) must be provided."
                )
            assert mhn_model.meta is not None \
                and mhn_model.meta.get("lambda") is not None, (
                    "MHN metadata is needed for MCMC sampling."
                    "Load a trained MHN model with metadata or manually "
                    "set mhn_model.meta['lambda'].")
            optimizer = oMHNOptimizer() if isinstance(mhn_model, oMHN) \
                else cMHNOptimizer()
            optimizer.load_data_matrix(data)
            optimizer._result = mhn_model
            optimizer._penalty = None

        else:
            if mhn_model is not None or data is not None:
                raise ValueError(
                    "Provide either optimizer or (mhn_model, data)"
                )
            assert optimizer.result is not None, (
                "Optimizer must be trained before passing to Sampler."
            )

            # TODO if penalty is not None:
            #     if penalty != optimizer.penalty:
            #         raise ValueError(
            #             "When providing a trained optimizer, do not provide "
            #             "a penalty."
            #         )
            penalty = optimizer._penalty

        # Transform penalty/prior into length-3 tuples

        if isinstance(penalty, Penalty):
            penalty = self._penalty_dict[type(optimizer)][penalty]

        if penalty is None:
            penalty = (None, None, None)
        penalty = tuple(penalty)
        if len(penalty) < 3:
            penalty = penalty + (None,) * (3 - len(penalty))
        self._penalty = penalty

        if log_prior is None:
            log_prior = (None, None, None)
        log_prior = tuple(log_prior)
        if len(log_prior) < 3:
            log_prior = log_prior + (None,) * (3 - len(log_prior))
        self._log_prior = log_prior

        # Set log_prior and its derivatives

        if (log_prior[0] is None) + (penalty[0] is None) != 1:
            raise ValueError(
                "Provide either penalty or log_prior, but not both."
            )
        self.log_prior = log_prior[0] or self._get_log_prior(
            penalty[0])

        if kernel_class in [MALAKernel, smMALAKernel]:
            if (log_prior[1] is None) + (penalty[1] is None) != 1:
                raise ValueError(
                    "Provide either gradient of penalty or gradient of "
                    "log_prior, but not both."
                )
            self.log_prior_grad = log_prior[1] or self._get_log_prior_grad(
                penalty[1])

        if kernel_class == smMALAKernel:
            if (log_prior[2] is None) + (penalty[2] is None) != 1:
                raise ValueError(
                    "Provide either Hessian of penalty or Hessian of "
                    "log_prior, but not both."
                )
            self.log_prior_hessian = log_prior[2] or \
                self._get_log_prior_hessian(penalty[2])

        self.optimizer = optimizer

        self.n_chains = n_chains
        self.size = optimizer.result.log_theta.size
        self.backup_interval = None
        self.backup_filename = None
        self.log_thetas = np.array([]).reshape(n_chains, 0, self.size)
        self.step_size = step_size
        self.thin = thin
        self.initial_step = None

        seed_sequence = np.random.SeedSequence(seed)
        self.rng = np.random.Generator(
            np.random.PCG64(seed_sequence.spawn(1)[0]))
        self.kernel_rngs = [
            np.random.Generator(
                np.random.PCG64(sese),
            )
            for sese in seed_sequence.spawn(self.n_chains)
        ]
        self.init_dist = None
        self.grad_and_log_likelihood = self._get_grad_and_log_likelihood()

        self.kernel_class = kernel_class
        if kernel_class in [MALAKernel, smMALAKernel]:
            self._log_prior_grad = None
        if kernel_class == smMALAKernel:
            self._log_prior_hessian = None

        self.n_samples = self.optimizer._data.get_data_shape()[0]
        self.lam = self.optimizer.result.meta["lambda"] * self.n_samples

        self.shape = optimizer.result.log_theta.shape

    def _get_grad_and_log_likelihood(
            self) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
        """
        Get the grad_and_log_likelihood function for an MHN. The default
        gradient and likelihood in the mhn.training module are
        normalized by the dataset size. This is reversed here
        """

        n_samples = self.optimizer._data.get_data_shape()[0]

        if isinstance(self.optimizer, oMHNOptimizer):

            def grad_and_log_likelihood(log_theta: np.ndarray) \
                    -> tuple[np.ndarray, float]:
                grad, log_likelihood = omhn_grad_and_log_likelihood(
                    omega_theta=log_theta.reshape(self.shape),
                    mutation_data=self.optimizer._data,
                )
                return n_samples * grad, n_samples * log_likelihood

        else:

            def grad_and_log_likelihood(log_theta: np.ndarray) \
                    -> tuple[np.ndarray, float]:
                grad, log_likelihood = cmhn_grad_and_log_likelihood(
                    log_theta=log_theta.flatten(),
                    data_matrix=self.optimizer._data,
                )
                return n_samples * grad, n_samples * log_likelihood

        return grad_and_log_likelihood

    def _get_log_prior(
        self, penalty: Callable[[np.ndarray], float]
    ) -> Callable[[np.ndarray], float]:
        """Get the log_prior as -lam * penalty, where lam is the
        regularization strength from MHN training.

        Args:
            penalty (Callable[[np.ndarray], float]): The penalty
            function used for MHN training.
        """

        def log_prior(log_theta: np.ndarray) -> float:
            return -self.lam * penalty(log_theta)

        return log_prior

    def _get_log_prior_grad(
        self, penalty_grad: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Get the log_prior_grad as -lam * penalty_grad, where lam is
        the regularization strength from MHN training.

        Args:
            penalty_grad (Callable[[np.ndarray], np.ndarray]): The
                gradient of the penalty function used for MHN training.
        """

        def log_prior_grad(log_theta: np.ndarray) -> float:
            return -self.lam * penalty_grad(log_theta)

        return log_prior_grad

    def _get_log_prior_hessian(
        self, penalty_hessian: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Get the log_prior_hessian as -lam * penalty_hessian, where
        lam is the regularization strength from MHN training.

        Args:
            penalty_hessian (Callable[[np.ndarray], np.ndarray]): The
                hessian of the penalty function used for MHN training.
        """

        def log_prior_hessian(log_theta: np.ndarray) -> float:
            return -self.lam * penalty_hessian(log_theta)

        return log_prior_hessian

    def _take_initial_step(self):
        """Take the initial step for MCMC sampling. Only works if
        penalty is one of the predefined penalties for
        ``mhn.training.optimizer``.

        Raises:
            NotImplementedError: When using a custom penalty, initial
                chain values must be set manually by setting
                ``Sampler.initial_step`` to an array of shape
                (n_chains, 1, m) with m the number of parameters.
        """
        if self.initial_step is not None:
            return

        if self._penalty[0] in [
            penalties_omhn.l2,
            penalties_cmhn.l2,
        ]:
            self.initial_step = self.rng.normal(
                size=(self.n_chains, 1, self.size),
                scale=1 / np.sqrt(2 * self.lam),
            )

        elif self._penalty[0] in [
            penalties_omhn.l1,
            penalties_omhn.sym_sparse,
            penalties_cmhn.l1,
            penalties_cmhn.sym_sparse,
        ]:
            self.initial_step = self.rng.laplace(
                size=(self.n_chains, 1, self.size),
                scale=1 / self.lam,
            )

        else:
            raise NotImplementedError(
                "When using a custom penalty, you must manually set " +
                "the initial chain values by setting " +
                R"`Sampler.initial_step` to an array of shape " +
                "(n_chains, 1, m) with m the number of parameters. "
            )

    def _walker(
        self,
        prev_step: np.ndarray,
        walker_id: int,
        n_steps: int,
        verbose: bool,
    ) -> tuple[int, np.ndarray, np.random.Generator]:
        """Internal walker function. Performs the MCMC sampling for one
        chain.

        Args:
            prev_step (np.ndarray): Initial value of one chain
            walker_id (int): ID of the chain.
            n_steps (int): Number of steps to run.
            verbose (bool): Whether to print progress.

        Returns:
            tuple[int, np.ndarray, np.random.Generator]: A tuple of the
            walker ID, the log_thetas sampled by this walker, and the
            random number generator state after sampling.
        """
        prev_n = self.log_thetas.shape[1]

        kernel = self.kernel_class(
            rng=self.kernel_rngs[walker_id],
            step_size=self.step_size if isinstance(
                self.step_size, float) else self.step_size[walker_id],
            grad_and_log_likelihood=self.grad_and_log_likelihood,
            log_prior=self.log_prior,
            shape=self.shape,
            **{arg: getattr(self, arg)
                for arg in self._kernel_args[self.kernel_class]},
        )

        log_thetas = np.empty((n_steps // self.thin, self.size))

        prev_step_res = kernel.get_params(prev_step)

        for r in range(n_steps):
            if verbose and walker_id == 0:
                print(
                    f"Step {prev_n * self.thin + r + 1:6}/{prev_n * self.thin + n_steps:6}", end="\r")

            prev_step, prev_step_res, _, _ = kernel.one_step(
                prev_step, prev_step_res, return_info=True
            )

            if (r + 1) % self.thin == 0:
                log_thetas[r // self.thin] = prev_step

        return walker_id, log_thetas, kernel.rng

    def _run(self, n_steps: int, verbose: bool):
        """Internal run function. Calls walker function in parallel.

        Args:
            n_steps (int): Number of steps to run.
            verbose (bool): Whether to print progress.
        """
        with mp.Pool() as pool:
            results = pool.starmap(
                self._walker,
                [(
                    self.log_thetas[i, -1, :] if self.log_thetas.shape[1] > 0
                    else self.initial_step[i, 0, :],
                    i, n_steps, verbose)
                 for i in range(self.n_chains)],
            )

        # Update kernel_rngs
        self.log_thetas = np.concatenate(
            [self.log_thetas, np.empty(
                (self.n_chains, n_steps // self.thin, self.size))],
            axis=1
        )

        for walker_id, log_thetas, kernel_rng in results:
            self.kernel_rngs[walker_id] = kernel_rng
            if n_steps // self.thin > 0:
                self.log_thetas[walker_id, -
                                (n_steps // self.thin):] = log_thetas

    def run(self,
            stopping_crit: Literal["r_hat", "ESS"] | Callable | None = "r_hat",
            max_steps: int | None = None, check_interval: int | None = 1000,
            burn_in: int | float = 0.2, verbose: bool = True) -> np.ndarray:
        """Run MCMC sampling until the stopping criterion is met or the
        maximum number of steps is reached.

        Args:
            stopping_crit (Literal["r_hat", "ESS"] | Callable | None, optional):
                The stopping criterion to use. If ``"r_hat"``, runs
                until the the Gelman-Rubin potential scale reduction
                factor :math:`\\hat R` is below 1.01. If ``"ESS"``, runs
                until the effective sample size (ESS) is above 100. If a
                callable, it should take in the log_thetas and return a
                boolean indicating whether to stop. If ``None``, runs
                until ``max_steps`` is reached. Burn-in is discarded
                before checking the stopping criterion and only every
                ``check_interval`` steps. Defaults to ``"r_hat"``.
            max_steps (int | None, optional): Maximum number of steps to
                run. If ``None``, runs a maximum of 1,000,000 steps for
                RWM and MALA kernels and 100,000 steps for smMALA
                kernels. Defaults to None.
            check_interval (int | None, optional): Number of steps
                between checking the stopping criterion. Defaults to
                1000.
            burn_in (int | float, optional): Number of steps to discard
                as burn-in. If a float, it is interpreted as a fraction
                of the total steps. Defaults to 0.2.
            verbose (bool, optional): Whether to print progress.
                Defaults to True.

        Returns:
            np.ndarray: The log-thetas for each chain.
        """
        if stopping_crit == "r_hat":
            def stopping_crit(log_thetas):
                if log_thetas.shape[1] < self.n_chains:
                    return False
                return np.all(
                    np.array(arviz.rhat(arviz.convert_to_dataset(log_thetas)
                                        ).to_array()) < 1.01)

        elif stopping_crit == "ESS":
            def stopping_crit(log_thetas):
                if log_thetas.shape[1] < self.n_chains:
                    return False
                return np.all(
                    np.array(arviz.ess(arviz.convert_to_dataset(log_thetas)
                                       ).to_array()) > 100)

        elif stopping_crit is None:
            def stopping_crit(log_thetas): return False

        max_steps = max_steps or (
            1_000_000 if self.kernel_class in [RWMKernel, MALAKernel]
            else 100_000 if self.kernel_class == smMALAKernel
            else None)

        max_steps = (max_steps // self.thin) * self.thin

        if isinstance(self.step_size, str) and self.step_size == "auto":
            if verbose:
                print("Tuning step size...")
            self.tune_stepsize()
            if verbose:
                print(f"Using step size: {self.step_size}")

        if self.log_thetas.shape[1] == 0:
            self._take_initial_step()

        if max_steps == 0:
            return self.log_thetas

        while max_steps and not stopping_crit(
            self.log_thetas[:, burn_in if isinstance(burn_in, int)
                            else int(burn_in * self.log_thetas.shape[1]):, :]):
            self._run(min(check_interval, max_steps),
                      verbose=verbose)
            max_steps -= min(check_interval, max_steps)

        return self.log_thetas

    def __getstate__(self):
        sampler = self.__dict__.copy()
        sampler.pop("grad_and_log_likelihood")
        sampler.pop("log_prior")
        sampler.pop("log_prior_grad", None)
        sampler.pop("log_prior_hessian", None)

        return sampler

    def acceptance(
            self, burn_in: int | float = 0.2, chain_id: int | None = None
    ) -> np.ndarray | float:
        """Calculate the acceptance rate for the chains.

        Args:
            burn_in (int | float, optional): Number of steps to discard
                as burn-in. If a float, it is interpreted as a fraction
                of the total steps. Defaults to 0.2.
            chain_id (int | None, optional): Chain ID to return
                acceptance rate for. If None, returns acceptance rates
                for all chains. Defaults to None.

        Returns:
            np.ndarray | float: The acceptance rate(s) for the specified
            chain(s).
        """
        if isinstance(burn_in, float):
            burn_in = int(burn_in * self.log_thetas.shape[1])

        log_thetas = self.log_thetas[:, burn_in:, :]

        acceptance_rates = list()

        for i in range(chain_id or log_thetas.shape[0]):
            accepted = np.sum(
                np.any(
                    log_thetas[i, 1:, :] != log_thetas[i, :-1, :],
                    axis=-1,
                )
            )
            total = log_thetas.shape[1] - 1
            acceptance_rates.append(accepted / total)

        return np.array(acceptance_rates) if chain_id is None \
            else acceptance_rates[0]

    def __setstate__(self, sampler):

        self.__dict__.update(sampler)
        self.grad_and_log_likelihood = self._get_grad_and_log_likelihood()
        self.log_prior = self._log_prior[0] or self._get_log_prior(
            self._penalty[0])
        if self.kernel_class in [MALAKernel, smMALAKernel]:
            self.log_prior_grad = self._log_prior[1] or self._get_log_prior_grad(
                self._penalty[1])
        if self.kernel_class == smMALAKernel:
            self.log_prior_hessian = self._log_prior[2] or \
                self._get_log_prior_hessian(self._penalty[2])

    def tune_stepsize(self, n_steps: int = 100, burn_in: float | int = 0.6,
                      target_acceptance: float | Literal["auto"] = "auto",
                      max_trials: int = 10, verbose: bool = True,
                      tol: float = 0.02,
                      ) -> float:
        """Automatically infer an appropriate step size epsilon for MCMC
        sampling.

        Args:
            n_steps (int, optional): Number of steps to run for
                inference. Defaults to 100.
            burn_in (float | int, optional): Burn-in period. If float,
                fraction of n_steps. If int, number of steps. Defaults
                to 0.6.
            target_acceptance (float | Literal["auto"], optional):
                Target acceptance rate. If "auto", set to 0.234 for RWM
                kernels, 0.574 for MALA kernels and 0.7 for smMALA
                kernels. Defaults to "auto".
            max_trials (int, optional): Maximum number of trials to run.
                Defaults to 10.
            verbose (bool, optional): Whether to print progress.
                Defaults to True.
            tol (float, optional): Tolerance for acceptance rate. If the
                acceptance rate is within ``tol`` of the target
                acceptance rate, the step size is accepted. Defaults to
                0.02.

        Returns:
            float: Inferred step size epsilon.
        """
        if target_acceptance == "auto":
            if self.kernel_class == RWMKernel:
                target_acceptance = 0.234
            elif self.kernel_class == MALAKernel:
                target_acceptance = 0.574
            elif self.kernel_class == smMALAKernel:
                target_acceptance = 0.7

        n_parallel = 5
        step_sizes = 10 ** np.linspace(-5, -1, n_parallel)

        for trial in range(max_trials):
            if verbose:
                print(f"Trial {trial+1}: step_sizes={step_sizes}")

            temp_sampler = MCMC(
                mhn_model=self.optimizer.result,
                data=self.optimizer.training_data,
                n_chains=n_parallel * 3,
                step_size=step_sizes.repeat(3),
                penalty=self._penalty,
                log_prior=self._log_prior,
                kernel_class=self.kernel_class,
                thin=1,
                seed=self.rng.integers(0, 2**32, dtype=np.uint32)
            )

            if self.initial_step is not None:
                temp_sampler.initial_step = np.stack(
                    [self.initial_step[i % self.n_chains, :, :]
                     for i in range(n_parallel * 3)])

            try:
                temp_sampler.run(max_steps=n_steps, verbose=True,
                                 stopping_crit=None)
            except ZeroDivisionError as e:
                if verbose:
                    print(f"Trial {trial+1} failed with error: {e}\n"
                          "Decreasing step sizes.")
                step_sizes *= 0.1
                continue

            acceptance_rates = temp_sampler.acceptance(
                burn_in=burn_in).reshape(n_parallel, 3).mean(axis=1)

            if verbose:
                print(f"Acceptance rates: {acceptance_rates}")

            argbest = np.argmin(np.abs(acceptance_rates - target_acceptance))
            if np.abs(acceptance_rates[argbest] - target_acceptance) < tol:
                self.step_size = step_sizes[argbest]
                return step_sizes[argbest]
            if acceptance_rates[argbest] < target_acceptance:
                step_sizes = np.linspace(
                    (step_sizes[argbest - 1] if argbest > 0
                     else step_sizes[argbest] / 10),
                    step_sizes[argbest],
                    n_parallel)
            else:
                step_sizes = np.linspace(
                    step_sizes[argbest],
                    (step_sizes[argbest + 1] if argbest < n_parallel - 1
                     else step_sizes[argbest] * 10),
                    n_parallel)

    def rhat(self, burn_in: int | float = 0.2, **kwargs) -> np.ndarray:
        """Calculate the Gelman-Rubin potential scale reduction factor
        :math:`\\hat R`.

        Args:
            burn_in (int | float, optional): Number of steps to discard
                as burn-in. If a float, it is interpreted as a fraction
                of the total steps. Defaults to 0.2.
            **kwargs: Additional keyword arguments to pass to
                :func:`arviz.rhat`. See the ArviZ documentation for more
                details.

        Returns:
            np.ndarray: The Gelman-Rubin R-hat values for each parameter.
        """
        if isinstance(burn_in, float):
            burn_in = int(burn_in * self.log_thetas.shape[1])

        log_thetas = self.log_thetas[:, burn_in:, :]

        return np.array(arviz.rhat(
            arviz.convert_to_inference_data(log_thetas), **kwargs).x)

    def ess(self, burn_in: int | float = 0.2, **kwargs):
        """Calculate the effective sample size (ESS).

        Args:
            burn_in (int | float, optional): Number of steps to discard
                as burn-in. If a float, it is interpreted as a fraction
                of the total steps. Defaults to 0.2.
            **kwargs: Additional keyword arguments to pass to
                :func:`arviz.ess`. See the ArviZ documentation for more
                details.

        Returns:
            np.ndarray: The effective sample size for each parameter.
        """
        if isinstance(burn_in, float):
            burn_in = int(burn_in * self.log_thetas.shape[1])

        log_thetas = self.log_thetas[:, burn_in:, :]

        return np.array(arviz.ess(
            arviz.convert_to_inference_data(log_thetas), **kwargs).x)
