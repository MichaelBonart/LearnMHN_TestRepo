# author: Y. Linda Hu

from ..optimizers import Optimizer, oMHNOptimizer, cMHNOptimizer, Penalty
from ..model import oMHN, cMHN
from ..training.state_containers import StateContainer
from ..training.likelihood_cmhn import gradient_and_score as cmhn_grad_and_log_likelihood
from ..training.likelihood_omhn import gradient_and_score as omhn_grad_and_log_likelihood
from numpy.typing import ArrayLike
from typing import Callable, Literal, overload
import numpy as np
import multiprocessing as mp
from .kernels import Kernel, smMALAKernel, RWMKernel, MALAKernel
from ..training import penalties_cmhn, penalties_omhn


class MCMC:
    """Markov chain Monte Carlo sampler for oMHN and cMHN models.

    Args:
        optimizer (Optimizer, optional): Trained Optimizer.
        mhn (oMHN | cMHN, optional): MHN model. Required if optimizer is not provided.
        data (ArrayLike | StateContainer, optional): Data used to train the MHN model.
            Required if optimizer is not provided.
        penalty (Penalty | tuple[Callable[[np.ndarray], float],
            Callable[[np.ndarray], np.ndarray]], optional): Penalty used during training.
            If not Penalty, penalty[0] gives the penalty (unscaled by lambda),
            penalty[1] its gradient and penalty[2] its Hessian.
            For a RWM kernel, only penalty[0] is required.
            For a MALA kernel, penalty[0] and penalty[1] are required.
            For a smMALA kernel, all three are required.
            If neither optimizer not penalty are provided, a log prior (and if applicable,)
            its derivatives have to be set manually with
            `Sampler.log_prior`, `Sampler.log_prior_grad`, and `Sampler.log_prior_hessian`.
        n_chains (int, optional): Number of parallel chains to run. Defaults to 10.
        epsilon (float | None | Literal["auto"], optional): Step size for MCMC sampler.
            If "auto", step size is set automatically inferred at the first run.
            Defaults to "auto".
        kernel_class (Kernel, optional): Kernel class to use for MCMC sampling.
            Defaults to MALAKernel.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        _type_: _description_
    """

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
    def __init__(self, *, optimizer: ..., n_chains: ... = ..., epsilon: ... = ...,
                 kernel_class: MALAKernel | RWMKernel = ..., seed: ... = ...,
                 ): ...

    @overload
    def __init__(self, *, mhn_model: ..., data: ...,
                 penalty: Penalty | Callable[[np.ndarray], float],
                 n_chains: ... = ..., epsilon: ... = ...,
                 kernel_class: Literal[RWMKernel] = ..., seed: ... = ...
                 ): ...

    @overload
    def __init__(self, *, mhn_model: ..., data: ...,
                 log_prior: Callable[[np.ndarray], float],
                 n_chains: ... = ..., epsilon: ... = ...,
                 kernel_class: Literal[RWMKernel] = ..., seed: ... = ...
                 ): ...

    @overload
    def __init__(self, *, mhn_model: ..., data: ...,
                 penalty: Penalty | tuple[
                     Callable[[np.ndarray], float],
                     Callable[[np.ndarray], np.ndarray]],
                 n_chains: ... = ..., epsilon: ... = ...,
                 kernel_class: Literal[MALAKernel] = ...,
                 seed: ... = ...): ...

    @overload
    def __init__(self, *, mhn_model: ..., data: ...,
                 log_prior: tuple[
                     Callable[[np.ndarray], float],
                     Callable[[np.ndarray], np.ndarray],],
                 n_chains: ... = ..., epsilon: ... = ...,
                 kernel_class: Literal[MALAKernel] = ...,
                 seed: ... = ...): ...

    @overload
    def __init__(self, *, mhn_model: ..., data: ...,
                 log_prior: tuple[
                     Callable[[np.ndarray], float],
                     Callable[[np.ndarray], np.ndarray],
                     Callable[[np.ndarray], np.ndarray],],
                 n_chains: ... = ..., epsilon: ... = ...,
                 kernel_class: Literal[smMALAKernel] = ...,
                 seed: ... = ...): ...

    def __init__(self, *, optimizer=None, mhn_model=None, data=None, penalty=None,
                 log_prior=None, n_chains=10, epsilon="auto",
                 kernel_class=MALAKernel, seed=0,) -> None:
        if optimizer is None:
            if mhn_model is None or data is None:
                raise ValueError(
                    "Either optimizer or (mhn_model, data) must be provided."
                )
            assert mhn_model.meta is not None and mhn_model.meta.get("lambda") is not None, (
                "MHN metadata is needed for MCMC sampling."
                "Load a trained MHN model with metadata or manually set"
                " mhn_model.meta['lambda'].")
            optimizer = oMHNOptimizer() if isinstance(mhn_model, oMHN) else cMHNOptimizer()
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

        # Transform penalty/prior into length-3 tuples

        if isinstance(penalty, Penalty):
            penalty = self.penalties[type(optimizer)][penalty]

        if penalty is None:
            penalty = (None, None, None)
        penalty = tuple(penalty)
        if len(penalty) < 3:
            penalty = penalty + (None,) * (3 - len(penalty))

        if log_prior is None:
            log_prior = (None, None, None)
        log_prior = tuple(log_prior)
        if len(log_prior) < 3:
            log_prior = log_prior + (None,) * (3 - len(log_prior))

        # Set log_prior and its derivatives
        
        if (log_prior[0] is None + penalty[0] is None) != 1:
            raise ValueError(
                "Provide either penalty or log_prior, but not both."
            )
        self.log_prior = log_prior[0] or self._get_log_prior(
            penalty[0])

        if kernel_class in [MALAKernel, smMALAKernel]:
            if (log_prior[1] is None + penalty[1] is None) != 1:
                raise ValueError(
                    "Provide either gradient of penalty or gradient of "
                    "log_prior, but not both."
                )
            self.log_prior_grad = log_prior[1] or self._set_log_prior_grad(
                penalty[1])

        if kernel_class == smMALAKernel:
            if (log_prior[2] is None + penalty[2] is None) != 1:
                raise ValueError(
                    "Provide either Hessian of penalty or Hessian of "
                    "log_prior, but not both."
                )
            self.log_prior_hessian = log_prior[2] or self._set_log_prior_hessian(
                penalty[2])

        self.optimizer = optimizer

        self.n_chains = n_chains
        self.size = optimizer.result.log_theta.size
        self.backup_interval = None
        self.backup_filename = None
        self.log_thetas = np.array([]).reshape(n_chains, 0, self.size)
        self.step_size = epsilon

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
        self._log_prior = None
        if kernel_class in [MALAKernel, smMALAKernel]:
            self._log_prior_grad = None
        if kernel_class == smMALAKernel:
            self._log_prior_hessian = None

        self.n_samples = self.optimizer._data.get_data_shape()[0]
        self.lam = self.optimizer.result.meta["lambda"] * self.n_samples

        self.shape = optimizer.result.log_theta.shape
        self.acceptance = np.zeros(n_chains)

    def _get_grad_and_log_likelihood(self):

        n_samples = self.optimizer._data.get_data_shape()[0]

        if isinstance(self.optimizer, oMHNOptimizer):

            def grad_and_log_likelihood(log_theta: np.ndarray) -> tuple[np.ndarray, float]:
                grad, log_likelihood = omhn_grad_and_log_likelihood(
                    omega_theta=log_theta.reshape(self.shape),
                    mutation_data=self.optimizer._data,
                )
                return n_samples * grad, n_samples * log_likelihood

        else:

            def grad_and_log_likelihood(log_theta: np.ndarray) -> tuple[np.ndarray, float]:
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
            return self.lam * penalty(log_theta)

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
            return self.lam * penalty_grad(log_theta)

        return log_prior_grad

    def _get_log_prior_hessian(
        self, penalty_hessian: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Get the log_prior_hessian as n_samples * lam * penalty_hessian,
        where lam is the regularization strength from MHN training.

        Args:
            penalty_hessian (Callable[[np.ndarray], np.ndarray]): The
            hessian of the penalty function used for MHN training.
        """

        def log_prior_hessian(log_theta: np.ndarray) -> float:
            return self.lam * penalty_hessian(log_theta)

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
    ):

        kernel = self.kernel_class(
            rng=self.kernel_rngs[walker_id],
            step_size=self.step_size,
            grad_and_log_likelihood=self.grad_and_log_likelihood,
            log_prior=self.log_prior,
            shape=self.shape,
            **{arg: getattr(self, arg) for arg in self.kernel_args[self.kernel_class]},
        )

        log_thetas = np.array([]).reshape(0, prev_step.shape[0])
        n_accepted = 0

        prev_step_res = kernel.get_params(prev_step)

        for r in range(n_steps):
            if walker_id == 0:
                print(f"Step {r:6}/{n_steps:6}", end="\r")

            prev_step, prev_step_res, ratio, accepted = kernel.one_step(
                prev_step, prev_step_res, return_info=True
            )
            log_thetas = np.vstack([log_thetas, prev_step])
            n_accepted += accepted

        return walker_id, log_thetas, n_accepted, kernel.rng

    def run(self, n_steps: int):

        n_before = self.log_thetas.shape[1]

        if n_before == 0 and n_steps > 0:
            self._take_initial_step()
            n_steps = n_steps - 1

        if n_steps == 0:
            return self.log_thetas

        with mp.Pool() as pool:
            results = pool.starmap(
                self.walker,
                [(prev_step, i, n_steps)
                 for i, prev_step in enumerate(self.log_thetas[:, -1, :])],
            )

        # Update kernel_rngs
        self.log_thetas = np.concatenate(
            [self.log_thetas, np.empty((self.n_chains, n_steps, self.size))],
            axis=1
        )
        for walker_id, log_thetas, n_accepted, kernel_rng in results:
            self.kernel_rngs[walker_id] = kernel_rng
            self.log_thetas[walker_id, -n_steps:] = log_thetas
            self.acceptance[walker_id] = (
                self.acceptance[walker_id] * n_before + n_accepted) / (n_before + n_steps)

        return self.log_thetas

    def __getstate__(self):
        sampler = self.__dict__.copy()
        sampler.pop("grad_and_log_likelihood")
        sampler.pop("log_prior")
        sampler.pop("log_prior_grad")

        return sampler

    def __setstate__(self, sampler):

        self.__dict__.update(sampler)
        self.grad_and_log_likelihood = self._get_grad_and_log_likelihood()
        self.log_prior = self._get_log_prior()
        self.log_prior_grad = self._get_log_prior_grad()
