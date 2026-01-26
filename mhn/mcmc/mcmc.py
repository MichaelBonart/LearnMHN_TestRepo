# author: Y. Linda Hu

from ..optimizers import Optimizer, oMHNOptimizer, Penalty
from ..model import oMHN, cMHN
from ..training.state_containers import StateContainer
from ..training.likelihood_cmhn import gradient_and_score as cmhn_grad_and_log_likelihood
from ..training.likelihood_omhn import gradient_and_score as omhn_grad_and_log_likelihood
from numpy.typing import ArrayLike
from typing import Callable, Literal
import numpy as np
import multiprocessing as mp
from .kernels import Kernel, smMALAKernel, RWMKernel, MALAKernel
from ..training import penalties_cmhn, penalties_omhn


class Sampler:

    kernel_args = {
        MALAKernel: ["log_prior_grad"],
        smMALAKernel: ["log_prior_grad", "log_prior_hessian"],
        RWMKernel: []
    }

    def __init__(
        self,
        optimizer: Optimizer = None,
        mhn: oMHN | cMHN = None,
        data: ArrayLike | StateContainer = None,
        penalty: Penalty | tuple[
            Callable[[np.ndarray], float],
            Callable[[np.ndarray], np.ndarray],
        ] = None,
        n_chains: int = 10,
        epsilon: float | None | Literal["auto"] = "auto",
        kernel_class: Kernel = MALAKernel,
        seed=0,
    ):
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
        self.log_prior = self._get_log_prior()
        self.log_prior_grad = self._get_log_prior_grad()
        self.n_samples = self.optimizer._data.get_data_shape()[0]
        self.lam = self.optimizer.result.meta["lambda"] * self.n_samples

        self.kernel_class = kernel_class
        self.shape = optimizer.result.log_theta.shape
        self.acceptance = np.zeros(n_chains)

    def _get_grad_and_log_likelihood(self):

        n_samples = self.optimizer._data.get_data_shape()[0]

        if isinstance(self.optimizer, oMHNOptimizer):

            def grad_and_log_likelihood(log_theta: np.ndarray) -> tuple[np.ndarray, float]:
                grad, log_likelihood = \
                    omhn_grad_and_log_likelihood(
                        omega_theta=log_theta.reshape(self.shape),
                        mutation_data=self.optimizer._data,
                    )
                return n_samples * grad, n_samples * log_likelihood

        else:

            def grad_and_log_likelihood(log_theta: np.ndarray) -> tuple[np.ndarray, float]:
                grad, log_likelihood = \
                    cmhn_grad_and_log_likelihood(
                        log_theta=log_theta.flatten(),
                        data_matrix=self.optimizer._data,
                    )
                return n_samples * grad, n_samples * log_likelihood

        return grad_and_log_likelihood

    def _get_log_prior(self):

        def log_prior(log_theta: np.ndarray) -> float:
            return self.lam * self.optimizer.penalty[0](log_theta)

        return log_prior

    def _get_log_prior_grad(self):

        def log_prior_grad(log_theta: np.ndarray) -> float:
            return self.lam * self.optimizer.penalty[1](log_theta)

        return log_prior_grad

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
