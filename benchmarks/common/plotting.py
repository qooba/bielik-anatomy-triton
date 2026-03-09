"""
Shared plotting utilities for benchmarks.

Provides reusable functions for creating consistent visualizations across all benchmarks.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "triton": "#2E86AB",
    "pytorch_native": "#A23B72",
    "pytorch_unfused": "#F18F01",
}


def plot_summary_comparison(
    data: Dict[str, List[float]],
    x_labels: List[str],
    metrics: List[str],
    title: str,
    xlabel: str,
    save_path: str,
    filename_prefix: str,
    gpu_name: str = "",
) -> None:
    """
    Create bar chart comparisons for multiple metrics.
    """
    provider_colors = []
    provider_labels = list(data.keys())

    for provider in provider_labels:
        provider_lower = provider.lower()
        if "triton" in provider_lower and "fused" in provider_lower:
            provider_colors.append(COLORS["triton"])
        elif "pytorch" in provider_lower and "native" in provider_lower:
            provider_colors.append(COLORS["pytorch_native"])
        elif "pytorch" in provider_lower and "unfused" in provider_lower:
            provider_colors.append(COLORS["pytorch_unfused"])
        elif "triton" in provider_lower and ("tensor" in provider_lower or "tc" in provider_lower):
            provider_colors.append(COLORS["triton"])
        elif "pytorch" in provider_lower and (
            "cublas" in provider_lower or "matmul" in provider_lower
        ):
            provider_colors.append(COLORS["pytorch_native"])
        elif "triton" in provider_lower and "basic" in provider_lower:
            provider_colors.append(COLORS["pytorch_unfused"])
        else:
            provider_colors.append("#888888")

    data_arrays = [np.array(values) for values in data.values()]
    num_providers = len(provider_labels)
    num_cases = len(x_labels)

    # Create plot for each metric
    for metric_idx, metric_name in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(num_cases)
        width = 0.8 / num_providers

        bars_list = []
        for i, (values, label, color) in enumerate(
            zip(data_arrays, provider_labels, provider_colors)
        ):
            offset = (i - num_providers / 2 + 0.5) * width

            if values.ndim > 1:
                metric_values = values[:, metric_idx]
            elif len(values) > 0 and isinstance(values[0], (list, tuple)):
                metric_values = [v[metric_idx] for v in values]
            else:
                metric_values = values

            bars = ax.bar(x + offset, metric_values, width, label=label, color=color)
            bars_list.append(bars)

        ylabel = metric_name
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")

        full_title = f"{title} - {metric_name}"
        if gpu_name:
            full_title += f"\nGPU: {gpu_name}"
        ax.set_title(full_title, fontsize=14, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    if height >= 1000:
                        label_text = f"{height:.0f}"
                    elif height >= 10:
                        label_text = f"{height:.1f}"
                    else:
                        label_text = f"{height:.2f}"

                    ax.annotate(
                        label_text,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        for bars in bars_list:
            autolabel(bars)

        plt.tight_layout()

        metric_suffix = metric_name.lower().replace("/", "_").replace(" ", "_")
        plot_file = Path(save_path) / f"{filename_prefix}-{metric_suffix}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved {metric_name} plot to: {plot_file}")


def create_summary_data(
    test_cases: List[tuple],
    providers: List[tuple],
    warmup_fn: Optional[callable] = None,
) -> tuple:
    """
    Helper to collect benchmark data for summary plots.
    """
    x_labels = []
    data_dict = {name: [] for name, _ in providers}

    for case_name, setup_args in test_cases:
        x_labels.append(case_name)

        if warmup_fn:
            warmup_fn(*setup_args)

        for provider_name, bench_fn in providers:
            result = bench_fn(*setup_args)
            data_dict[provider_name].append(result)

    return x_labels, data_dict
