#!/usr/bin/env python3
"""Causal Inference in Time Series Econometrics."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from src.core import (
    difference_in_differences,
    perform_granger_causality_test,
    plot_causal_effect,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal Inference in Time Series")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["output"]["figures_dir"])
    )
    output_dir.mkdir(exist_ok=True)

    if config["data"]["generate_synthetic"]:
        np.random.seed(config["data"]["seed"])
        n = config["data"]["n_periods"]
        x = np.cumsum(np.random.normal(0, 1, n))
        y = 0.5 * x + np.random.normal(0, 0.5, n)
        data = pd.DataFrame({"x": x, "y": y})
    else:
        raise ValueError("No data source specified")

    if config["analysis"]["granger_causality"]["enabled"]:
        perform_granger_causality_test(
            data, "x", "y", config["analysis"]["granger_causality"]["maxlag"]
        )

    if config["analysis"]["difference_in_differences"]["enabled"]:
        n = config["data"]["n_periods"]
        y_treated = np.cumsum(np.random.normal(0.1, 1, n))
        y_control = np.cumsum(np.random.normal(0, 1, n))
        treatment_period = config["analysis"]["difference_in_differences"][
            "treatment_period"
        ]
        did = difference_in_differences(y_treated, y_control, treatment_period)
        logging.info(f"Difference-in-differences estimate: {did:.4f}")
        plot_causal_effect(
            y_treated,
            y_control,
            treatment_period,
            output_dir / "causal_effect.png",
            plot=True,
        )

    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
