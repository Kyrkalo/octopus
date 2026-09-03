# Getting Started

`octopus` is a config-driven ML-ops engine: you install it as a regular
Python dependency, implement a handful of contracts in your own project,
point a `configs.json` at them, and run everything through one CLI. This
page gets you from zero to a first successful run.

## Requirements

- Python >= 3.9
- A project of your own to install `octopus` into (it's a dependency,
  not a template — see [Home](Home) for how the pieces fit together)

## Install

Pick whichever fits how you're consuming it:

```bash
# Local editable checkout — for developing octopus itself, or a
# path dependency while you iterate on both repos side by side
pip install -e /path/to/octopus

# Directly from GitHub — the common case for a consuming project
pip install git+https://github.com/Kyrkalo/octopus.git
```

To pin it as a dependency in another project's `pyproject.toml`:

```toml
dependencies = [
    "octopus @ git+https://github.com/Kyrkalo/octopus.git",
]
```

`main` has no tagged releases yet, so a plain `git+https://...` install
always tracks the latest commit on `main`. Pin `@<tag>` once releases
exist.

## Verify the install

```bash
python -c "import octopus; print(octopus.__file__)"
python -m octopus --help
```

`python -m octopus --help` should print the `list` / `run` / `run-all`
subcommands. If `import octopus` fails, double-check you installed into
the Python environment/venv you're actually running from.

## Your first run

`octopus` ships with an empty `configs.json` (just `dataset_root`) — real
model configs live in your project, not the engine. The fastest way to
see it work end-to-end is the `ThreeAddX` toy model on the
[CLI Usage](CLI-Usage) page: four small classes plus one `configs.json`
entry that trains a model to learn `3 + x = 5`. Follow that page's
"Example: ThreeAddX model" section, then:

```bash
python -m octopus --config configs.json list           # shows: ThreeAddX
python -m octopus --config configs.json run ThreeAddX
```

## Next steps

- [CLI Usage](CLI-Usage) — every subcommand, plus the full `ThreeAddX` walkthrough
- [Base Contracts](Base-Contracts) — the classes you implement (`BaseDataLoader`, `BasePipeline`, `BaseTrainer`, `BaseTester`, `Exporter`; models are plain `torch.nn.Module`)
- [Configuration](Configuration) — how `configs.json` and `dataset_root` resolution work
- [PipelineMetricsCollector](PipelineMetricsCollector) — tracking metrics from inside a pipeline
- [Home](Home) — full wiki map

## Related pages

- [Home](Home)
- [CLI Usage](CLI-Usage)
