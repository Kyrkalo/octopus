from abc import ABC, abstractmethod


class BaseTester(ABC):
    """Contract every tester implementation must follow: evaluate the
    model for one epoch and report back whatever stats the pipeline logs."""

    @abstractmethod
    def test(self, epoch: int):
        raise NotImplementedError
