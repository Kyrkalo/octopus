# octopus

## What is octopus

`octopus` is a reusable, config-driven ML-ops package. It wires together the
usual training pieces — dataloader, model, pipeline, trainer, tester,
exporter — without ever hardcoding which concrete classes belong to which
model. Everything is resolved at runtime from a JSON config, so adding a new
model means writing an implementation and a config block, not editing
`octopus` itself.

It's a standalone, installable package meant to be a dependency of your
training project, not a template you copy into it.

## Introduction

`octopus` grew out of a single monolithic training script that hardcoded
every model it supported. That doesn't scale past a handful of models, and
it means every new model requires touching shared code. `octopus` splits
that into two halves:

- **The engine** (this repo, `octopus/`) — abstract contracts for each
  moving part, a config loader, and a factory that dynamically imports and
  wires concrete classes by dotted path. This half never changes per-model
  and has no knowledge of any consuming project.
- **Your implementations** — concrete dataloaders/models/pipelines/etc.
  that follow those contracts, living in your own project (wherever you
  want, as long as it's importable), plus a `configs.json` that points at
  them by dotted path.

The name follows the same idea: `octopus` is the body with many
independent, pluggable arms (dataloaders, models, trainers, ...), each one
your own implementation, coordinated from one shared core. That core - the
config loader and factory - lives in `octopus/platypus/`.

## Install

`octopus` is a regular installable Python package (not a template to copy
into your project). From your consuming project:

```bash
pip install -e /path/to/octopus          # local editable checkout
# or
pip install git+https://github.com/Kyrkalo/octopus.git   # pinned/remote
```

Then `import octopus` works from anywhere in your project, and
`python -m octopus run|run-all|list --config path/to/your/configs.json`
runs your models.

## Explore concepts

```
octopus/
  platypus/          # the engine: config.py (load configs.json) + factory.py (ModelFactory)
  metrics/            # pipeline/epoch metrics collection, used by BasePipeline
  utils/              # shared torchvision/COCO helpers + GPT checkpoint tools
  dataloaders/base.py # BaseDataLoader   - implement setup()
  pipelines/base.py   # BasePipeline     - implement setup() / run()
  trainers/base.py    # BaseTrainer      - implement train(epoch)
  testers/base.py     # BaseTester       - implement test(epoch)
  exporters/base.py   # Exporter         - implement setup() / run()
  configs.json         # one block per model key
  cli.py / __main__.py # `python -m octopus run|run-all|list`
```

**`configs.json` is the single source of truth.** Each top-level key is a
model you can run. Its `components` block maps role → dotted class path;
everything else in the block is that model's hyperparameters. This repo
ships `configs.json` with no model blocks — it only exists to define
`dataset_root` — because those blocks belong to your project, not the
engine. Point `--config` at your own file, e.g.:

```json
"mnist": {
  "components": {
    "dataloader": "myproject.mnist.dataloader.MnistDataLoader",
    "model":      "myproject.mnist.model.MnistModel",
    "pipeline":   "myproject.mnist.pipeline.MnistPipeline",
    "trainer":    "myproject.mnist.trainer.MnistTrainer",
    "tester":     "myproject.mnist.tester.MnistTester",
    "exporter":   "myproject.mnist.exporter.MnistExportOnnx"
  },
  "learning_rate": 0.01,
  "n_epochs": 1
}
```

**`ModelFactory`** (`octopus/platypus/factory.py`) reads a config key,
imports `components.pipeline`, constructs it with the config, and calls
`pipeline.setup().run()`. If `components.exporter` is present it does the
same for the exporter afterwards, passing `pipeline.model`. That's the
entire wiring mechanism - no registry, no enum of known models.

## Build

To add a model, implement the pieces you need against these contracts
(only `dataloader`, `pipeline`, and `trainer` are required by the factory
- `dataset`, `tester`, and `exporter` are optional per model):

| Base | You implement | Contract |
|---|---|---|
| `BaseDataLoader` | `setup(self)` | build `self.dataset` / `self.data_loader`; `get()` calls `setup()` and returns `self.data_loader` |
| *(model)* | `forward(self, x)` | plain `torch.nn.Module` - octopus has no base class for models |
| `BasePipeline` | `setup(self)`, `run(self)` | `setup()` must return `self` and set `self.model`; also gives you metrics/checkpoint helpers (`start_pipeline`, `end_epoch`, `load_pretrained_model`, ...) |
| `BaseTrainer` | `train(self, epoch)` | runs one training epoch |
| `BaseTester` | `test(self, epoch)` | runs one evaluation epoch |
| `Exporter` | `setup(self)`, `run(self)` | takes `(config, model=...)`; base gives you `getPath(extension)` |

Put the implementation in a self-contained folder in your own project
(e.g. `myproject/<model>/`: `dataloader.py`, `model.py`, `pipeline.py`,
`trainer.py`, ...) anywhere importable - `octopus` never imports it
directly, only your `configs.json` points at it.

## Write your first flow

A minimal "toy" model that trains `y = 3x + 1` on random data, built in
your own project (which has `octopus` installed as a dependency).

**1. `myproject/toy/dataloader.py`**
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from octopus.dataloaders.base import BaseDataLoader

class ToyDataLoader(BaseDataLoader):
    def setup(self):
        x = torch.randn(200, 1)
        y = 3 * x + 1
        self.dataset = TensorDataset(x, y)
        self.data_loader = DataLoader(self.dataset, batch_size=self.config["batch_size"], shuffle=True)
```

**2. `myproject/toy/model.py`**
```python
import torch.nn as nn

class ToyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
```

**3. `myproject/toy/trainer.py`**
```python
import torch.nn as nn
from octopus.trainers.base import BaseTrainer

class ToyTrainer(BaseTrainer):
    def __init__(self, model, optimizer, loader):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.criterion = nn.MSELoss()

    def train(self, epoch):
        for x, y in self.loader:
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            loss.backward()
            self.optimizer.step()
        print(f"epoch {epoch}  loss {loss.item():.4f}")
```

**4. `myproject/toy/pipeline.py`**
```python
import torch.optim as optim
from octopus.pipelines.base import BasePipeline
from myproject.toy.dataloader import ToyDataLoader
from myproject.toy.model import ToyLinear
from myproject.toy.trainer import ToyTrainer

class ToyPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self):
        loader = ToyDataLoader(self.config).get()
        self.model = ToyLinear()
        optimizer = optim.SGD(self.model.parameters(), lr=self.config["lr"])
        self.trainer = ToyTrainer(self.model, optimizer, loader)
        return self

    def run(self):
        for epoch in range(1, self.config["num_epochs"] + 1):
            self.trainer.train(epoch)
        return self
```

**5. Add a block to your own `configs.json`** (not the one bundled with
this repo — see [Install](#install))
```json
"toy": {
  "components": {
    "dataloader": "myproject.toy.dataloader.ToyDataLoader",
    "model":      "myproject.toy.model.ToyLinear",
    "pipeline":   "myproject.toy.pipeline.ToyPipeline",
    "trainer":    "myproject.toy.trainer.ToyTrainer"
  },
  "batch_size": 32,
  "lr": 0.01,
  "num_epochs": 20
}
```

**6. Run it**
```bash
python -m octopus run toy --config path/to/your/configs.json
```

No exporter, no dataset class, no edits anywhere under `octopus/` - that's
the whole point.

## Development

To work on `octopus` itself:

```bash
pip install -e ".[dev]"
pytest
```

The test suite (`tests/`) covers the engine only — `platypus.config` and
`platypus.factory` — using fake pipeline/exporter classes, so it has no
dependency on any consuming project's model code.
