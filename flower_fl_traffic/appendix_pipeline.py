"""Command-line entry point for the paper's supplementary experiments.

The centralized saturation curve and unequal-client-data experiment are
available here without changing or extending the main FL pipeline.
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Dispatch an appendix workflow and forward its options to that workflow.

    Args:
        argv: Optional command-line argument list, primarily for testing.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        usage=(
            "%(prog)s {saturation,imbalance} "
            "[--mode {run,rerun,plot}] [workflow options]"
        ),
    )
    parser.add_argument(
        "workflow",
        choices=("saturation", "imbalance"),
        help=(
            "'saturation' measures centralized accuracy in 5%% increments; "
            "'imbalance' compares local and federated learning across unequal "
            "client data ratios."
        ),
    )

    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return

    workflow = arguments[0]
    if workflow not in {"saturation", "imbalance"}:
        parser.error(f"invalid workflow: {workflow!r}")

    if workflow == "saturation":
        from utils.central_baseline import main as run_workflow
    else:
        from utils.multi_split import main as run_workflow

    run_workflow(arguments[1:])


if __name__ == "__main__":
    main()
