# CLI Usage

`octopus` ships a small CLI (`python -m octopus`) for running whatever
models are wired up in your project's `configs.json`. It has three
subcommands: `list`, `run`, and `run-all`.

```
usage: python -m octopus [-h] [--config PATH] {run,run-all,list} ...
```

`--config PATH` points at your project's `configs.json` (see
[Example: ThreeAddX model](#example-threeaddx-model) below). If omitted,
it falls back to the empty `configs.json` bundled with the `octopus`
package itself, which has no model blocks.

## Commands

| Command | Description |
|---|---|
| `list` | Print every model key present in the config |
| `run <key>` | Run a single model by its config key |
| `run-all` | Run every model key present in the config, in order |

## Example: ThreeAddX model

`octopus` never ships real model configs — those belong to your project
(see [PipelineMetricsCollector](PipelineMetricsCollector) for how a
pipeline reports back into this system). Below is a complete, tiny model
called `ThreeAddX` that learns the equation `3 + x = 5`: it trains a
one-parameter linear model to predict `x + 3`, so that at `x = 2` it
outputs `~5`.

**1. `myproject/three_add_x/dataloader.py`**
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from octopus.dataloaders.base import BaseDataLoader

class ThreeAddXDataLoader(BaseDataLoader):
    def setup(self):
        x = torch.randn(200, 1)
        y = x + 3.0  # target: 3 + x
        self.dataset = TensorDataset(x, y)
        self.data_loader = DataLoader(self.dataset, batch_size=self.config["batch_size"], shuffle=True)
```

**2. `myproject/three_add_x/model.py`**
```python
import torch.nn as nn

class ThreeAddXModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # learns y = w*x + b, ideally w=1, b=3

    def forward(self, x):
        return self.linear(x)
```

**3. `myproject/three_add_x/trainer.py`**
```python
import torch.nn as nn
from octopus.trainers.base import BaseTrainer

class ThreeAddXTrainer(BaseTrainer):
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

**4. `myproject/three_add_x/pipeline.py`**
```python
import torch
import torch.optim as optim
from octopus.pipelines.base import BasePipeline
from myproject.three_add_x.dataloader import ThreeAddXDataLoader
from myproject.three_add_x.model import ThreeAddXModel
from myproject.three_add_x.trainer import ThreeAddXTrainer

class ThreeAddXPipeline(BasePipeline):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self):
        loader = ThreeAddXDataLoader(self.config).get()
        self.model = ThreeAddXModel()
        optimizer = optim.SGD(self.model.parameters(), lr=self.config["lr"])
        self.trainer = ThreeAddXTrainer(self.model, optimizer, loader)
        return self

    def run(self):
        for epoch in range(1, self.config["num_epochs"] + 1):
            self.trainer.train(epoch)

        with torch.no_grad():
            prediction = self.model(torch.tensor([[2.0]]))
        print(f"3 + 2 ~= {prediction.item():.2f}")
        return self
```

**5. `configs.json`**
```json
{
  "ThreeAddX": {
    "components": {
      "dataloader": "myproject.three_add_x.dataloader.ThreeAddXDataLoader",
      "model":      "myproject.three_add_x.model.ThreeAddXModel",
      "pipeline":   "myproject.three_add_x.pipeline.ThreeAddXPipeline",
      "trainer":    "myproject.three_add_x.trainer.ThreeAddXTrainer"
    },
    "batch_size": 32,
    "lr": 0.01,
    "num_epochs": 20
  }
}
```

**6. Run it**

```bash
python -m octopus --config configs.json list           # shows: ThreeAddX
python -m octopus --config configs.json run ThreeAddX   # run one model
python -m octopus --config configs.json run-all         # run everything
```

`run` exits with an error (and lists the available keys) if you pass a
key that isn't in the config:

```bash
$ python -m octopus --config configs.json run not_a_real_model
Unknown model key: 'not_a_real_model'. Available: ThreeAddX
```

## Related pages

- [Home](Home)
- [Getting Started](Getting-Started)
- [Configuration](Configuration)
- [PipelineMetricsCollector](PipelineMetricsCollector)
