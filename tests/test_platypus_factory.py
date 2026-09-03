import pytest

from octopus.platypus.factory import ModelFactory


class FakePipeline:
    ran = False

    def __init__(self, config):
        self.config = config
        self.model = "fake-model"

    def setup(self):
        return self

    def run(self):
        FakePipeline.ran = True


class FakeExporter:
    exported_with = None

    def __init__(self, config, model=None):
        self.config = config
        self.model = model

    def setup(self):
        return self

    def run(self):
        FakeExporter.exported_with = self.model


@pytest.fixture(autouse=True)
def reset_fakes():
    FakePipeline.ran = False
    FakeExporter.exported_with = None
    yield


def test_execute_unknown_key_raises_value_error():
    factory = ModelFactory(configs={})

    with pytest.raises(ValueError, match="Unknown model key"):
        factory.execute("does-not-exist")


def test_execute_missing_pipeline_component_raises_value_error():
    factory = ModelFactory(configs={"mnist": {"components": {}}})

    with pytest.raises(ValueError, match="missing components.pipeline"):
        factory.execute("mnist")


def test_execute_runs_pipeline_and_exporter(monkeypatch):
    configs = {
        "mnist": {
            "components": {
                "pipeline": "tests.test_platypus_factory.FakePipeline",
                "exporter": "tests.test_platypus_factory.FakeExporter",
            }
        }
    }
    factory = ModelFactory(configs=configs)

    result = factory.execute("mnist")

    assert result.config is configs["mnist"]
    assert FakePipeline.ran is True
    assert FakeExporter.exported_with == "fake-model"


def test_execute_without_exporter_component_skips_export(monkeypatch):
    configs = {
        "mnist": {
            "components": {
                "pipeline": "tests.test_platypus_factory.FakePipeline",
            }
        }
    }
    factory = ModelFactory(configs=configs)

    factory.execute("mnist")

    assert FakePipeline.ran is True
    assert FakeExporter.exported_with is None


def test_run_all_executes_every_key(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ModelFactory, "execute", lambda self, key: calls.append(key)
    )
    factory = ModelFactory(configs={"a": {}, "b": {}})

    factory.run_all()

    assert calls == ["a", "b"]
