import argparse
import sys

from octopus.platypus.config import load_configs
from octopus.platypus.factory import ModelFactory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m octopus",
        description="Run a config-driven training pipeline.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to configs.json (defaults to the bundled octopus/configs.json)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single model by its config key")
    run_parser.add_argument("key", help='Config key to run, e.g. "mnist" or "cnn14_2_cbam"')

    subparsers.add_parser("run-all", help="Run every model key present in the config")

    subparsers.add_parser("list", help="List the model keys available in the config")

    return parser


def _dispatch(args, configs: dict) -> int:
    factory = ModelFactory(configs)

    if args.command == "list":
        for key in configs:
            print(key)
        return 0

    if args.command == "run":
        if args.key not in configs:
            print(f"Unknown model key: {args.key!r}. Available: {', '.join(configs)}", file=sys.stderr)
            return 1
        factory.execute(args.key)
        return 0

    if args.command == "run-all":
        factory.run_all()
        return 0

    return 1


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    configs = load_configs(args.config) if args.config else load_configs()

    try:
        return _dispatch(args, configs)
    except KeyboardInterrupt:
        print("\nInterrupted - process terminated.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
