"""octopus - a reusable, config-driven ML-ops package.

Every pluggable part (dataloaders, datasets, pipelines, trainers, testers,
exporters) is an "arm": you implement your own concrete versions and wire
them together through a JSON config, without touching the shared core in
``octopus.platypus``. Models are plain ``torch.nn.Module`` subclasses -
octopus has no base class for them.

Run a model:
    python -m octopus run <model_key>

See ``custom`` for reference implementations built on top of this
package, and ``octopus.platypus.factory`` for how a config drives a run.
"""

__all__ = [
    "platypus",
    "metrics",
    "utils",
    "dataloaders",
    "datasets",
    "exporters",
    "pipelines",
    "testers",
    "trainers",
]
