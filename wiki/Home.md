# octopus wiki

`octopus` is a reusable, config-driven ML-ops engine for PyTorch training
pipelines. This wiki is the reference layer under the README: the README
covers "what is this and how do I start," the wiki covers "here's every
piece, in depth," and it should grow one page at a time as the engine
grows.

## How this wiki is organized

Each page maps to one piece of the engine, mirroring the `octopus/`
package layout. When you add or change functionality, update the page
for the folder you touched — create it from the matching row below if it
doesn't exist yet, and add the row/sidebar entry if you're introducing a
new one.

| `octopus/` folder or file | Wiki page | Status |
|---|---|---|
| (project root: install, quickstart) | [Getting Started](Getting-Started) | done |
| `cli.py`, `__main__.py` | [CLI Usage](CLI-Usage) | done |
| `platypus/config.py`, `configs.json` | [Configuration](Configuration) | done |
| `platypus/factory.py` + overall design | [Architecture Overview](Architecture-Overview) | to write |
| `dataloaders/`, `pipelines/`, `trainers/`, `testers/`, `exporters/` (the `Base*` contracts — models are plain `nn.Module`, no octopus base class) | [Base Contracts](Base-Contracts) | to write |
| `metrics/collector.py` | [PipelineMetricsCollector](PipelineMetricsCollector) | done |
| `metrics/metric_entry.py`, `epoch_summary.py`, `pipeline_run.py`, `enums.py` | [Metrics Data Model](Metrics-Data-Model) | to write |
| `utils/` (coco_eval, coco_utils, engine, transforms, common, gpt) | [Utils Reference](Utils-Reference) | to write |
| how to add a new model end-to-end | [Writing a New Model](Writing-A-New-Model) | to write |
| dev setup, running tests, PR expectations | [Contributing](Contributing) | to write |

"Status" is just a running note — flip it to `done` (or delete the row
entirely once every page exists) as you fill pages in. Don't let this
table drift from reality: if a page stops matching the code, that's a
signal to update the page, not to leave it stale.

## Conventions for new pages

- One page per row above; don't fold unrelated modules into one page as
  the engine grows — split instead.
- Add every new page to [_Sidebar](_Sidebar) in the matching section (or
  a new section if it's a new category of concept).
- Cross-link related pages under a trailing `## Related pages` section
  (see [PipelineMetricsCollector](PipelineMetricsCollector) or
  [CLI Usage](CLI-Usage) for the pattern).
- Keep runnable examples actually runnable — verify code snippets against
  the real engine before publishing them, the way the `ThreeAddX` example
  on [CLI Usage](CLI-Usage) was tested end-to-end.
