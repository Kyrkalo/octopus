"""The engine: config loading and the generic ModelFactory.

Everything else shared (metrics, utils) lives as a sibling package under
``octopus``; this module is deliberately just the two moving parts that
turn a config entry into a running pipeline.

Usage:
    from octopus.platypus.factory import run
    run("cnn14")
"""

__all__ = ["config", "factory"]
