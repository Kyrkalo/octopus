from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    """Contract every dataloader implementation must follow.

    Subclasses implement ``setup()``: build ``self.dataset`` and
    ``self.data_loader`` from ``self.config``. ``get()`` runs ``setup()``
    and returns ``self.data_loader`` - callers only ever need to call
    ``get()``.
    """

    def __init__(self, config: dict):
        self.config = config
        self.dataset = None
        self.data_loader = None

    @abstractmethod
    def setup(self):
        """Build self.dataset and self.data_loader from self.config."""
        raise NotImplementedError

    def get(self):
        """Run setup() and return the resulting data loader."""
        self.setup()
        return self.data_loader
