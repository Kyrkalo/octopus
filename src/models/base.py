import torch.nn as nn


class BaseModel(nn.Module):
    """Thin nn.Module wrapper shared by every model implementation.

    Subclasses still implement ``forward`` themselves like any other
    ``nn.Module`` - this base just carries the utility methods pipelines
    rely on across every model.
    """

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
