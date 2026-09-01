from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Contract every trainer implementation must follow: run one epoch
    of training and report back whatever stats the pipeline logs."""

    @abstractmethod
    def train(self, epoch: int):
        raise NotImplementedError
