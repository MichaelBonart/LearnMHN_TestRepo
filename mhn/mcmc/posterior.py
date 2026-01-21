"""
This submodule contains functionalities to use the Metropolis-adjusted Langevin algorithm (MALA) for sampling from posterior distributions of MHNs.
"""
# author: Y. Linda Hu

import pints
from ..optimizers import Optimizer, oMHNOptimizer
from .. import training
import numpy as np


class MHNLikelihood(pints.LogPDF):

    def __init__(self, optimizer: Optimizer):
        """
        Initializes the MHNLikelihood with the given MHN optimizer.

        :param optimizer: An instance of mhn.optimizers.Optimizer
        """
        self.data = optimizer._data
        self.n_samples = self.data.get_data_shape()[0]
        self.omhn = isinstance(optimizer, oMHNOptimizer)
        self.gradient_likelihood = \
            training.likelihood_omhn.gradient_and_score \
            if self.omhn\
            else training.likelihood_cmhn.gradient_and_score

        n_events = optimizer._data.get_data_shape()[1]
        self._x = n_events + 1 if self.omhn else n_events
        self._y = n_events
        self._n_parameters = self._x * self._y

    def evaluateS1(self, log_theta):
        """
        Evaluates the log-likelihood and its gradient at the given log_theta.

        :param log_theta: Description
        """

        grad, log_likelihood = self.gradient_likelihood(
            log_theta.reshape(self._x, self._y), self.data)
        return self.n_samples * log_likelihood, self.n_samples * grad.flatten()

    def n_parameters(self):
        return self._n_parameters


class MHNPrior(pints.LogPrior):

    def __init__(self, optimizer: Optimizer):
        """
        Initializes the MHNPrior with the given MHN optimizer.

        :param optimizer: An instance of mhn.optimizers.Optimizer
        """

        if optimizer._penalty is None or optimizer.result is None:
            raise ValueError(
                "MHN has not been trained yet.")
        self.prior, self.gradient = optimizer._penalty
        self.lam = optimizer._data.get_data_shape(
        )[0] * optimizer.result.meta["lambda"]
        self.omhn = isinstance(optimizer, oMHNOptimizer)

        n_events = optimizer._data.get_data_shape()[1]
        self._x = n_events + 1 if self.omhn else n_events
        self._y = n_events
        self._n_parameters = self._x * self._y

    def evaluateS1(self, log_theta):
        """
        Evaluates the log-prior and its gradient at the given log_theta.

        :param log_theta: Description
        """

        return (- self.lam * self.prior(log_theta.reshape(self._x, self._y)),
                - self.lam * self.gradient(log_theta.reshape(self._x, self._y)).flatten())

    def n_parameters(self):
        return self._n_parameters

    def draw(self, n: int = 1) -> np.ndarray:
        """
        Draws a sample from the prior distribution.

        :return: A sample from the prior.
        """
        if self.prior in [
            training.penalties_cmhn.l1,
            training.penalties_cmhn.sym_sparse,
            training.penalties_omhn.l1,
            training.penalties_omhn.sym_sparse
        ]:
            return np.random.laplace(
                scale=1.0 / self.lam,
                size=(n, self._x * self._y)
            )
        elif self.prior in [
            training.penalties_cmhn.l2,
            training.penalties_omhn.l2
        ]:
            return np.random.normal(
                size=(n, self._x * self._y),
                scale=1 / np.sqrt(2 * self.lam)
            )
        else:
            raise NotImplementedError(
                "Sampling from this prior is not implemented.")


class MHNPosterior(pints.LogPosterior):

    def __init__(self, optimizer: Optimizer):
        """
        Initializes the MHNPosterior with the given MHN optimizer.

        :param optimizer: An instance of mhn.optimizers.Optimizer
        """

        self.likelihood = MHNLikelihood(optimizer)
        self.prior = MHNPrior(optimizer)
        super().__init__(self.likelihood, self.prior)
