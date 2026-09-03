# PipelineMetricsCollector

`octopus.metrics.collector.PipelineMetricsCollector` is the centralized
metrics collector for ML pipelines built on `octopus`. It gives every
`BasePipeline` a unified interface for tracking metrics across different
kinds of training runs and saving them in a structured JSON format.

## When to use it

Any `BasePipeline` implementation that wants epoch/metric tracking,
best-metric bookkeeping, or a JSON run record should go through this
class rather than rolling its own logging. `BasePipeline.setup_metrics()`
constructs one for you — see [BasePipeline](BasePipeline) — but it can
also be used standalone via `create_collector(...)`.

## Constructor

```python
PipelineMetricsCollector(
    pipeline_name: str,
    pipeline_type: str,
    run_id: str | None = None,
    output_dir: str = "mlops_data/metrics",
    config: dict | None = None,
    auto_save: bool = True,
)
```

| Argument | Description |
|---|---|
| `pipeline_name` | Name of the pipeline (e.g. `"MNIST_Training"`, `"CNN14_AudioTagging"`) |
| `pipeline_type` | Type of pipeline (e.g. `"supervised_classification"`, `"gan_training"`) |
| `run_id` | Unique run identifier; auto-generated from a timestamp if omitted |
| `output_dir` | Directory metrics JSON files are written to |
| `config` | Pipeline configuration dict; non-JSON-serializable values (e.g. `torch.device`) are sanitized automatically |
| `auto_save` | If `True`, `end_epoch()` writes the metrics JSON after every epoch |

On construction, the collector also captures environment info
(Python/PyTorch version, CUDA availability, GPU name, etc.) and stores it
on the run record.

## Usage example

```python
# Initialize collector
collector = PipelineMetricsCollector(
    pipeline_name="MNIST_Training",
    pipeline_type="supervised_classification",
    config={"epochs": 20, "batch_size": 32}
)

# Start pipeline
collector.start_pipeline()

# Training loop
for epoch in range(num_epochs):
    collector.start_epoch(epoch)

    # Training phase
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = train_step(data, target)
        collector.log_metric("train_loss", loss.item(), step=batch_idx)

    # Validation phase
    val_metrics = validate()
    collector.log_phase_metrics("val", val_metrics)

    # End epoch
    collector.end_epoch(epoch, {
        "train_loss": avg_train_loss,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"]
    })

# Save model
collector.add_artifact("model_path", "models/mnist_model.pth")
collector.add_artifact("onnx_path", "models/mnist_model.onnx")

# End pipeline
collector.end_pipeline(
    status=PipelineStatus.COMPLETED,
    final_metrics={"best_val_accuracy": 0.98}
)
```

## Method reference

| Method | Purpose |
|---|---|
| `start_pipeline(additional_config=None)` | Marks the run as `RUNNING` and stamps the start time; optionally merges extra config |
| `start_epoch(epoch)` | Begins timing a new epoch |
| `end_epoch(epoch, epoch_metrics=None)` | Records epoch duration and metrics, updates best-metric tracking, and (if `auto_save`) writes the JSON file |
| `log_metric(name, value, step=None, epoch=None, metric_type=MetricType.SCALAR, metadata=None)` | Logs one metric value |
| `log_phase_metrics(phase, metrics)` | Logs a dict of metrics under a phase prefix (`train`/`val`/`test`) |
| `log_batch_metrics(batch_idx, metrics)` | Logs a dict of per-batch metrics |
| `add_artifact(artifact_name, artifact_path)` | Records a produced artifact (checkpoint, ONNX export, ...) |
| `end_pipeline(status=PipelineStatus.COMPLETED, final_metrics=None, error_message=None, error_traceback=None)` | Marks the run finished (or failed), computes total duration, and does a final save |
| `save_to_json(filename=None)` | Writes the full run record to `output_dir/filename` (auto-named if omitted) |
| `get_summary()` | Returns a dict summary: status, duration, best/final metrics, artifacts |
| `get_metric_history(metric_name)` | Returns every logged entry for one metric name |
| `get_epoch_summary(epoch)` | Returns the recorded summary for one epoch, or `None` |

### Best-metric tracking

`end_epoch()` automatically tracks the best value seen so far for every
numeric metric: names containing `"loss"` or `"error"` are treated as
lower-is-better, everything else (accuracy, precision, ...) as
higher-is-better. Access the result via `collector.get_summary()["best_metrics"]`.

## Factory function

```python
from octopus.metrics.collector import create_collector

collector = create_collector("MNIST_Training", "supervised_classification")
```

`create_collector(pipeline_name, pipeline_type, config=None, output_dir="mlops_data/metrics")`
is a thin convenience wrapper around the constructor above.

## Related pages

- [BasePipeline](BasePipeline)
- [Metrics data model](Metrics-Data-Model) — `MetricEntry`, `EpochSummary`, `PipelineRun`
