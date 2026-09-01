
import os
import torch
from octopus.metrics import PipelineMetricsCollector
from octopus.metrics.enums import PipelineStatus


class BasePipeline:
    def __init__(self):
        self.model = None
        self.config = None
        self.metrics_collector = None
        pass

    def setup_metrics(self, name: str, type: str, dir: str):
        self.metrics_collector = PipelineMetricsCollector(
            pipeline_name=name,
            pipeline_type=type,
            config=self.config,
            output_dir=dir
        )
        pass

    def start_pipeline(self) -> None:
        self.metrics_collector.start_pipeline()

    def start_epoch(self, epoch: int) -> None:
        self.metrics_collector.start_epoch(epoch)

    def add_artifact(self, name: str, path: str) -> None:
        self.metrics_collector.add_artifact(name, path)

    def end_pipeline(self, status: PipelineStatus, e: Exception) -> None:
        import traceback
        self.metrics_collector.end_pipeline(
                status=status,
                error_message=str(e),
                error_traceback=traceback.format_exc()
            )
        
    def end_pipeline(self, status: PipelineStatus, final_metrics) -> None:
        import traceback
        self.metrics_collector.end_pipeline(status=status, final_metrics= final_metrics)

    def log_phase_metrics(self, key: str, metrics: dict) -> None:
        self.metrics_collector.log_phase_metrics(key, metrics)

    def end_epoch(self, epoch: int, metrics: dict) -> None:
        self.metrics_collector.end_epoch(epoch, metrics)


    def count_parameters(self):
        """Count total trainable parameters in the model."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def load_pretrained_model(self, model, config):
        """
        Load pretrained model weights if path is provided in config.

        Args:
            model: The model instance to load weights into
            config: Configuration dict that may contain 'pretrain' key with model path

        Returns:
            The model with loaded weights (if applicable)
        """
        if not config.get("pretrain"):
            return model

        pretrain_path = config["pretrain"]

        if not os.path.exists(pretrain_path):
            print(f"Warning: Pretrain path specified but file not found: {pretrain_path}")
            return model

        print(f"Loading pretrained model from: {pretrain_path}")

        try:
            checkpoint = torch.load(pretrain_path, map_location=config.get("device", "cpu"))

            # Handle both checkpoint dict and direct state dict formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])

                # Display checkpoint metadata if available
                if 'val_acc' in checkpoint:
                    print(f"  Loaded model with validation accuracy: {checkpoint['val_acc']:.2f}%")
                if 'epoch' in checkpoint:
                    print(f"  Trained for {checkpoint['epoch']} epochs")
            else:
                # Direct state dict format
                model.load_state_dict(checkpoint)

            print("Pretrained model loaded successfully")

        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("  Continuing with fresh model initialization...")

        return model