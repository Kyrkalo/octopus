# octopus

## What is octopus

`octopus` is a reusable, config-driven ML-ops package. It wires together the
usual training pieces — dataloader, model, pipeline, trainer, tester,
exporter — without ever hardcoding which concrete classes belong to which
model. Everything is resolved at runtime from a JSON config, so adding a new
model means writing an implementation and a config block, not editing
`octopus` itself.

It's designed to be dropped into any project (or eventually extracted into
its own repo) and reused as-is.

## Introduction

`octopus` grew out of a single monolithic training script that hardcoded
every model it supported. That doesn't scale past a handful of models, and
it means every new model requires touching shared code. `octopus` splits
that into two halves:

- **The engine** (`octopus/`) — abstract contracts for each moving part,
  a config loader, and a factory that dynamically imports and wires
  concrete classes by dotted path. This half never changes per-model.
- **Your implementations** — concrete dataloaders/models/pipelines/etc.
  that follow those contracts, living wherever you want (this repo keeps
  its own under `src/custom/`).

The name follows the same idea: `octopus` is the body with many
independent, pluggable arms (dataloaders, models, trainers, ...), each one
your own implementation, coordinated from one shared core. That core - the
config loader and factory - lives in `octopus/platypus/`.

## Explore concepts

```
octopus/
  platypus/          # the engine: config.py (load configs.json) + factory.py (ModelFactory)
  metrics/            # pipeline/epoch metrics collection, used by BasePipeline
  utils/              # shared torchvision/COCO helpers + GPT checkpoint tools
  dataloaders/base.py # BaseDataLoader   - implement get()
  models/base.py      # BaseModel        - nn.Module + count_parameters()
  pipelines/base.py   # BasePipeline     - implement setup() / run()
  trainers/base.py    # BaseTrainer      - implement train(epoch)
  testers/base.py     # BaseTester       - implement test(epoch)
  exporters/base.py   # Exporter         - implement setup() / run()
  configs.json         # one block per model key
  cli.py / __main__.py # `python -m octopus run|run-all|list`
```

**`configs.json` is the single source of truth.** Each top-level key is a
model you can run. Its `components` block maps role → dotted class path;
everything else in the block is that model's hyperparameters:

```json
"mnist": {
  "components": {
    "dataloader": "src.custom.mnist.dataloader.MnistDataLoader",
    "model":      "src.custom.mnist.model.Mdl_mnist_202520",
    "pipeline":   "src.custom.mnist.pipeline.MnistPipeline",
    "trainer":    "src.custom.mnist.trainer.MNISTTrainer",
    "tester":     "src.custom.mnist.tester.MNISTTester",
    "exporter":   "src.custom.mnist.exporter.MnistExportOnnx"
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
| `BaseDataLoader` | `get(self)` | returns loader(s), e.g. `(train_loader, val_loader)` |
| `BaseModel` | `forward(self, x)` | plain `nn.Module`, plus free `count_parameters()` |
| `BasePipeline` | `setup(self)`, `run(self)` | `setup()` must return `self` and set `self.model`; also gives you metrics/checkpoint helpers (`start_pipeline`, `end_epoch`, `load_pretrained_model`, ...) |
| `BaseTrainer` | `train(self, epoch)` | runs one training epoch |
| `BaseTester` | `test(self, epoch)` | runs one evaluation epoch |
| `Exporter` | `setup(self)`, `run(self)` | takes `(config, model=...)`; base gives you `getPath(extension)` |

Put the implementation in a self-contained folder (mirroring
`src/custom/<model>/`: `dataloader.py`, `model.py`, `pipeline.py`,
`trainer.py`, ...) anywhere importable - `octopus` never imports it
directly, only `configs.json` points at it.

## Write your first flow

A minimal "toy" model that trains `y = 3x + 1` on random data.

**1. `src/custom/toy/dataloader.py`**
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from octopus.dataloaders.base import BaseDataLoader

class ToyDataLoader(BaseDataLoader):
    def get(self):
        x = torch.randn(200, 1)
        y = 3 * x + 1
        return DataLoader(TensorDataset(x, y), batch_size=self.config["batch_size"], shuffle=True)
```

**2. `src/custom/toy/model.py`**
```python
import torch.nn as nn
from octopus.models.base import BaseModel

class ToyLinear(BaseModel):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
```

**3. `src/custom/toy/trainer.py`**
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

**4. `src/custom/toy/pipeline.py`**
```python
import torch.optim as optim
from octopus.pipelines.base import BasePipeline
from src.custom.toy.dataloader import ToyDataLoader
from src.custom.toy.model import ToyLinear
from src.custom.toy.trainer import ToyTrainer

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

**5. Add a block to `octopus/configs.json`**
```json
"toy": {
  "components": {
    "dataloader": "src.custom.toy.dataloader.ToyDataLoader",
    "model":      "src.custom.toy.model.ToyLinear",
    "pipeline":   "src.custom.toy.pipeline.ToyPipeline",
    "trainer":    "src.custom.toy.trainer.ToyTrainer"
  },
  "batch_size": 32,
  "lr": 0.01,
  "num_epochs": 20
}
```

**6. Run it**
```bash
python -m octopus run toy
```

No exporter, no dataset class, no edits anywhere under `octopus/` - that's
the whole point.
