# author: Y. Linda Hu

from ..optimizers import Optimizer, oMHNOptimizer, cMHNOptimizer, Penalty
from ..model import oMHN, cMHN
from ..training.likelihood_cmhn \
    import gradient_and_score as cmhn_grad_and_log_likelihood
from ..training.likelihood_omhn \
    import gradient_and_score as omhn_grad_and_log_likelihood
from numpy.typing import ArrayLike
from typing import Callable, Literal, overload
import numpy as np
import multiprocessing as mp
from .kernels import Kernel, smMALAKernel, RWMKernel, MALAKernel
from ..training import penalties_cmhn, penalties_omhn
import arviz


class MCMC:
    """Markov chain Monte Carlo sampler for oMHN and cMHN models.

    Args:
        optimizer (Optimizer, optional): Trained Optimizer.
        mhn (oMHN | cMHN, optional): MHN model. Required if optimizer is 
            not provided.
        data (ArrayLike | StateContainer, optional): Data used to train
            the MHN model. Required if optimizer is not provided.
        penalty (Penalty | tuple[Callable[[np.ndarray], float],
            Callable[[np.ndarray], np.ndarray]], optional): Penalty used
            during training. If not Penalty, penalty[0] gives the
            penalty (unscaled by lambda), penalty[1] its gradient and 
            penalty[2] its Hessian. For a RWM kernel, only penalty[0] is
            required. For a MALA kernel, penalty[0] and penalty[1] are
            required. For a smMALA kernel, all three are required. If
            neither optimizer not penalty are provided, a log prior (and
            if applicable,) its derivatives have to be set manually with
            `Sampler.log_prior`, `Sampler.log_prior_grad`, and
            `Sampler.log_prior_hessian`.
        n_chains (int, optional): Number of parallel chains to run.
            Defaults to 10.
        epsilon (float | None | Literal["auto"], optional): Step size
            for MCMC sampler. If "auto", step size is set automatically
            inferred at the first run. Defaults to "auto".
        kernel_class (Kernel, optional): Kernel class to use for MCMC
            sampling. Defaults to MALAKernel.
        seed (int, optional): Random seed for reproducibility. Defaults
            to 0.

    Returns:
        _type_: _description_
    """

    import arviz

    kernel_args = {
        MALAKernel: ["log_prior_grad"],
        smMALAKernel: ["log_prior_grad", "log_prior_hessian"],
        RWMKernel: []
    }

    penalties = {
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

    @overload
    def __init__(self, *, optimizer: ..., n_chains: ... = ...,
                 step_size: ... = ...,
                 kernel_class: MALAKernel | RWMKernel = ..., thin: ... = ...,
                 seed: ... = ...,): ...

    @overload
    def __init__(self, *, mhn_model: ..., data: ...,
                 penalty: Penalty | Callable[[np.ndarray], float],
                 n_chains: ... = ..., step_size: ... = ...,
                 kernel_class: Literal[RWMKernel] = ..., thin: ... = ...,
                 seed: ... = ...): ...

    @overload
    def __init__(self, *, mhn_model: ..., data: ...,
                 log_prior: Callable[[np.ndarray], float],
                 n_chains: ... = ..., step_size: ... = ...,
                 kernel_class: Literal[RWMKernel] = ..., thin: ... = ...,
                 seed: ... = ...): ...

    @overload
    def __init__(self, *, mhn_model: ..., data: ...,
                 penalty: Penalty | tuple[
                     Callable[[np.ndarray], float],
                     Callable[[np.ndarray], np.ndarray]],
                 n_chains: ... = ..., step_size: ... = ...,
                 kernel_class: Literal[MALAKernel] = ..., thin: ... = ...,
                 seed: ... = ...): ...

    @overload
    def __init__(self, *, mhn_model: ..., data: ...,
                 log_prior: tuple[
                     Callable[[np.ndarray], float],

                     Callable[[np.ndarray], np.ndarray],],
                 n_chains: ... = ..., step_size: ... = ...,
                 kernel_class: Literal[MALAKernel] = ..., thin: ... = ...,
                 seed: ... = ...): ...

    @overload
    def __init__(self, *, mhn_model: ..., data: ...,
                 log_prior: tuple[
                     Callable[[np.ndarray], float],
                     Callable[[np.ndarray], np.ndarray],
                     Callable[[np.ndarray], np.ndarray],],
                 n_chains: ... = ..., step_size: ... = ...,
                 kernel_class: Literal[smMALAKernel] = ..., thin: ... = ...,
                 seed: ... = ...): ...

    def __init__(self, *, optimizer: Optimizer = None,
                 mhn_model: oMHN | cMHN | None = None, data=None, penalty=None,
                 log_prior=None, n_chains=10,
                 step_size: Literal["auto"] | float | ArrayLike = "auto",
                 kernel_class=MALAKernel, thin: int = 100, seed=0,) -> None:
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
            penalty = self.penalties[type(optimizer)][penalty]

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

    def _get_grad_and_log_likelihood(self):

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
        """Get the log_prior as n_samples * lam * penalty, where lam is
        the regularization strength from MHN training.

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
        """Get the log_prior_grad as n_samples * lam * penalty_grad,
        where lam is the regularization strength from MHN training.

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
        """Get the log_prior_hessian as
        n_samples * lam * penalty_hessian, where lam is the
        regularization strength from MHN training.

        Args:
            penalty_hessian (Callable[[np.ndarray], np.ndarray]): The
            hessian of the penalty function used for MHN training.
        """

        def log_prior_hessian(log_theta: np.ndarray) -> float:
            return -self.lam * penalty_hessian(log_theta)

        return log_prior_hessian

    def _take_initial_step(self):

        if self.optimizer.penalty[0] in [
            penalties_omhn.l2,
            penalties_cmhn.l2,
        ]:
            self.log_thetas = self.rng.normal(
                size=(self.n_chains, 1, self.size),
                scale=1 / np.sqrt(2 * self.lam),
            )

        elif self.optimizer.penalty[0] in [
            penalties_omhn.l1,
            penalties_omhn.sym_sparse,
            penalties_cmhn.l1,
            penalties_cmhn.sym_sparse,
        ]:
            self.log_thetas = self.rng.laplace(
                size=(self.n_chains, 1, self.size),
                scale=1 / self.lam,
            )

        else:
            raise NotImplementedError(
                "When using a custom penalty, you must manually set " +
                "the initial chain values by setting " +
                R"`Sampler.log_thetas` to an array of shape " +
                "(n_chains, 1, m) with m the number of parameters. "
            )

    def walker(
        self,
        prev_step: np.ndarray,
        walker_id: int,
        n_steps: int,
        verbose: bool,
        first_step_done: bool
    ):
        prev_n = self.log_thetas.shape[1]
        if first_step_done and prev_n == 1:
            prev_n = 0

        kernel = self.kernel_class(
            rng=self.kernel_rngs[walker_id],
            step_size=self.step_size if isinstance(
                self.step_size, float) else self.step_size[walker_id],
            grad_and_log_likelihood=self.grad_and_log_likelihood,
            log_prior=self.log_prior,
            shape=self.shape,
            **{arg: getattr(self, arg)
                for arg in self.kernel_args[self.kernel_class]},
        )

        log_thetas = np.empty((n_steps // self.thin, self.size))

        prev_step_res = kernel.get_params(prev_step)

        for r in range(n_steps):
            if verbose and walker_id == 0:
                print(
                    f"Step {prev_n * self.thin + r + 1 + first_step_done:6}/{prev_n * self.thin + n_steps + first_step_done:6}", end="\r")

            prev_step, prev_step_res, _, _ = kernel.one_step(
                prev_step, prev_step_res, return_info=True
            )
            if (r + first_step_done) % self.thin == 0:
                log_thetas[r // self.thin] = prev_step

        return walker_id, log_thetas, kernel.rng

    def _run(self, n_steps: int, verbose: bool, first_step_done: bool):

        with mp.Pool() as pool:
            results = pool.starmap(
                self.walker,
                [(prev_step, i, n_steps, verbose, first_step_done)
                 for i, prev_step in enumerate(self.log_thetas[:, -1, :])],
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
            max_steps: int | None = None, check_interval: int | None = None,
            burn_in: int | float = 0.2, verbose: bool = True):

        if stopping_crit == "r_hat":
            def stopping_crit(log_thetas):
                return \
                    np.all(np.array(arviz.rhat(arviz.convert_to_dataset(log_thetas)
                                               ).to_array()) < 1.01)

        elif stopping_crit == "ESS":
            def stopping_crit(log_thetas): return \
                np.all(np.array(arviz.ess(arviz.convert_to_dataset(log_thetas)
                                          ).to_array()) > 100)

        max_steps = max_steps or (
            1_000_000 if self.kernel_class in [RWMKernel, MALAKernel]
            else 100_000 if self.kernel_class == smMALAKernel
            else None)

        check_interval = check_interval or 1000

        if isinstance(self.step_size, str) and self.step_size == "auto":
            if verbose:
                print("Tuning step size...")
            self.tune_stepsize()
            if verbose:
                print(f"Using step size: {self.step_size}")

        n_before = self.log_thetas.shape[1]

        if max_steps == 0:
            return self.log_thetas

        if n_before == 0 and max_steps > 0:
            self._take_initial_step()

            if max_steps == 1:
                return self.log_thetas

            self._run(min(check_interval - 1, max_steps - 1),
                      verbose=verbose, first_step_done=True)
            max_steps -= min(check_interval, max_steps)

        while max_steps and not stopping_crit(
            self.log_thetas[:, burn_in if isinstance(burn_in, int)
                            else int(burn_in * self.log_thetas.shape[1]):, :]):
            self._run(min(check_interval, max_steps),
                      verbose=verbose, first_step_done=False)
            max_steps -= min(check_interval, max_steps)

        return self.log_thetas

    def __getstate__(self):
        sampler = self.__dict__.copy()
        sampler.pop("grad_and_log_likelihood")
        sampler.pop("log_prior")
        sampler.pop("log_prior_grad", None)
        sampler.pop("log_prior_hessian", None)

        return sampler

    def acceptance(self, burn_in: int | float = 0.2, chain_id: int | None = None):
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
                      max_trials: int = 10, verbose: bool = True, tol: float = 0.02,
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
            )

            temp_sampler.run(max_steps=n_steps, verbose=True)

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

    def rhat(self, burn_in: int | float = 0.2, **kwargs):
        if isinstance(burn_in, float):
            burn_in = int(burn_in * self.log_thetas.shape[1])

        log_thetas = self.log_thetas[:, burn_in:, :]

        return np.array(arviz.rhat(
            arviz.convert_to_inference_data(log_thetas), **kwargs).x)

    def ess(self, burn_in: int | float = 0.2, **kwargs):
        if isinstance(burn_in, float):
            burn_in = int(burn_in * self.log_thetas.shape[1])

        log_thetas = self.log_thetas[:, burn_in:, :]

        return np.array(arviz.ess(
            arviz.convert_to_inference_data(log_thetas), **kwargs).x)
