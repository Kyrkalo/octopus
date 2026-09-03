# Configuration

Everything `octopus` runs is described by one `configs.json` file, loaded
by `octopus.platypus.config.load_configs()`. This page covers its shape,
what `load_configs()` does to it before your pipeline ever sees it, and a
couple of gotchas worth knowing.

## Shape

```json
{
  "dataset_root": "../dataset",

  "<model_key>": {
    "components": {
      "dataloader": "dotted.path.to.YourDataLoader",
      "model":      "dotted.path.to.YourModel",
      "pipeline":   "dotted.path.to.YourPipeline",
      "trainer":    "dotted.path.to.YourTrainer",
      "tester":     "dotted.path.to.YourTester",
      "exporter":   "dotted.path.to.YourExporter"
    },
    "...": "any other hyperparameters your pipeline reads from config[...]"
  }
}
```

- Every top-level key **except** `dataset_root` is a **model key** —
  something you can pass to `python -m octopus run <key>`, and something
  `list` / `run-all` iterate over. See [CLI Usage](CLI-Usage).
- `components` is the only structural field `ModelFactory` reads. Only
  `pipeline` is required by the factory itself (it needs one to
  instantiate and call `setup().run()`); `dataloader`, `model`,
  `trainer`, `tester`, and `exporter` are conventions your own pipeline
  code is expected to look up from `components` and import itself — see
  [Base Contracts](Base-Contracts) and [Architecture Overview](Architecture-Overview)
  for how the factory actually wires things together.
- Everything else in a model's block is free-form: your `BasePipeline`,
  `BaseTrainer`, etc. read whatever keys they need straight off the
  `config` dict they're constructed with.

## `load_configs()`

```python
from octopus.platypus.config import load_configs

configs = load_configs("path/to/configs.json")   # or load_configs() for the bundled default
```

For each call, `load_configs()`:

1. Reads and parses the JSON file.
2. Pops the top-level `dataset_root` key out of the result entirely — it
   never appears in what your pipeline receives, and if it's absent it
   defaults to `"."`.
3. Resolves `dataset_root` to an **absolute path**, relative to the
   parent directory of wherever the installed `octopus` package lives on
   disk (two directories up from `platypus/config.py`) — **not** relative
   to your `configs.json` file, and **not** relative to your current
   working directory. In an editable/source checkout this is the repo
   root; in a normal `pip install`, it's the parent of `octopus/` inside
   `site-packages`. Keep `dataset_root` as an absolute path if you want
   it to behave predictably regardless of install method.
4. For every model block, replaces the literal substring `${dataset_root}`
   inside any **top-level string value** with that resolved absolute
   path. Substitution is not recursive — a `${dataset_root}` placeholder
   nested inside a dict or list value in a model block is left untouched.
5. Stamps `config["device"] = torch.device("cuda" if available else "cpu")`
   onto every model block. This is a real `torch.device` object, not a
   string — if you're logging a config that contains it (e.g. via
   [PipelineMetricsCollector](PipelineMetricsCollector)), that's exactly
   why its `_sanitize_config` step converts `torch.device` to a string
   before writing JSON.

## Default config path

`--config` is optional. Omit it and `load_configs()` falls back to
`octopus/configs.json`, the file bundled with the installed package
itself — which ships with no model blocks, just `{"dataset_root": "../dataset"}`.
Real configs belong to your project; point `--config` at your own file:

```bash
python -m octopus --config path/to/your/configs.json list
```

## Example

```json
{
  "dataset_root": "../dataset",

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

(This is the same `ThreeAddX` model walked through end-to-end on
[CLI Usage](CLI-Usage).)

## Gotchas

- **`dataset_root` resolution is install-location-relative, not
  cwd-relative.** If a fresh `pip install git+...` moves where the
  `octopus` package sits on disk, any relative `dataset_root` resolves
  differently than it did in your editable checkout. Prefer an absolute
  path if this matters for your setup.
- **Placeholder substitution is shallow.** Only string values directly on
  a model's top-level config dict get `${dataset_root}` replaced —
  nested structures don't.
- **`device` is injected, not something you set.** Don't put a `"device"`
  key in your own config blocks; `load_configs()` overwrites it
  unconditionally.

## Related pages

- [Home](Home)
- [Getting Started](Getting-Started)
- [CLI Usage](CLI-Usage)
- [Architecture Overview](Architecture-Overview)
- [PipelineMetricsCollector](PipelineMetricsCollector)
