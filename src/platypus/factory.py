import importlib

from octopus.platypus.config import load_configs


def _import_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class ModelFactory:
    """Reads configs.json and runs whatever model key is asked for.

    There is no hardcoded registry of model types: any key present in the
    config is runnable, as long as its "components.pipeline" (and
    optionally "components.exporter") dotted paths point at real classes.
    That's what makes this reusable per user - adding a new model means
    adding a config block + implementation, not editing this factory.
    """

    def __init__(self, configs: dict = None):
        self.configs = configs if configs is not None else load_configs()

    def execute(self, key: str):
        if key not in self.configs:
            raise ValueError(f"Unknown model key: {key!r} (not found in configs.json)")

        config = self.configs[key]
        components = config.get("components", {})
        if "pipeline" not in components:
            raise ValueError(f"Config {key!r} is missing components.pipeline")

        pipeline_class = _import_class(components["pipeline"])
        pipeline = pipeline_class(config)
        pipeline.setup().run()

        if "exporter" in components:
            exporter_class = _import_class(components["exporter"])
            exporter = exporter_class(config, model=pipeline.model)
            exporter.setup().run()

        return pipeline

    def run_all(self):
        for key in self.configs:
            self.execute(key)


def run(key: str):
    """Convenience entrypoint: ModelFactory().execute(key)."""
    return ModelFactory().execute(key)
