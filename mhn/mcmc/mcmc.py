# author: Y. Linda Hu

from ..optimizers import Optimizer, Penalty
from ..model import oMHN, cMHN
from ..training.state_containers import StateContainer
from numpy.typing import ArrayLike
from typing import Callable, Literal
import numpy as np
import multiprocessing as mp
from .kernels import Kernel, MALAKernel

WARN = {
    "INIT_DIST": True,
}


class Sampler:

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
        
        self.n_chains = n_chains        
        self.size = optimizer.result.log_theta.size
        self.backup_interval = None
        self.backup_filename = None
        self.log_thetas = np.array([]).reshape(0, self.size)

    def walker(
        self,
        prev_step: np.ndarray,
        walker_id: int,
        n_steps: int

    ):

        kernel = self.kernel_class(
            data=self.data,
            step_size=self.step_size,
            prior=self.prior,
            omhn=self.omhn,
            rng=self.kernel_rngs[walker_id],
        )

        log_thetas = np.array([]).reshape(0, prev_step.shape[0])

        prev_step_res = kernel.get_params(prev_step)

        for r in range(n_steps):
            if walker_id == 0:
                print(f"Step {r:6}/{n_steps:6}", end="\r")

            prev_step, prev_step_res, ratio, accepted = kernel.one_step(
                prev_step, prev_step_res, return_info=True
            )

            # if self.backup_interval is not None and r % self.backup_interval == 0:
            #     with NpyAppendArray(f"{self.backup_filename}_{walker_id}.npy") as npaa:
            #         npaa.append(log_thetas[-r:])

        return walker_id, log_thetas, kernel.rng


    def run(self, n_steps: int):

        with mp.Pool() as pool:
            results = pool.starmap(
                self.walker,
                [(i, n_steps) for i in range(self.n_chains)],
            )

        # Update kernel_rngs
        self.log_thetas = np.vstack(
            [self.log_thetas, np.empty((self.n_chains, n_steps, self.size))]
        )
        for walker_id, log_thetas, kernel_rng in results:
            self.kernel_rngs[walker_id] = kernel_rng
            self.log_thetas[walker_id, -n_steps:] = log_thetas

        self.n_steps += n_steps

        return self.log_theta