from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    """Contract every dataloader implementation must follow.

    A dataloader wraps whatever Dataset(s) a model needs and hands back
    ready-to-use ``torch.utils.data.DataLoader`` instances (or an
    equivalent iterable) for the pipeline to iterate over.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def get(self):
        """Return the loader(s) for this model, e.g. (train_loader, val_loader)."""
        raise NotImplementedError
