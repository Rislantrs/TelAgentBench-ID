"""Generate Figures 2, 3, and 4 from standardized TelAgentBench-ID evaluation metrics."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 300,
    }
)

# Standardized 12-model evaluation summary corresponding to Table 7 in the manuscript.
BENCHMARK_DATA = [
    {
        "Category": "Proprietary Cloud",
        "Model": "Claude-Opus-5",
        "ST_ToolAcc": 89.09,
        "ST_StrictAST": 87.27,
        "MT_StrictAST": 64.42,
        "Macro_StrictAST": 75.85,
        "ArgMatch": 87.27,
        "MacroF1": 78.40,
        "FormatValid": 99.01,
        "Latency": 3.94,
        "Throughput": 28.40,
        "VRAM": None,
        "OnPremise": False,
    },
    {
        "Category": "Proprietary Cloud",
        "Model": "MiniMax-M3",
        "ST_ToolAcc": 87.27,
        "ST_StrictAST": 85.45,
        "MT_StrictAST": 21.88,
        "Macro_StrictAST": 53.67,
        "ArgMatch": 85.45,
        "MacroF1": 61.20,
        "FormatValid": 97.03,
        "Latency": 5.43,
        "Throughput": 18.20,
        "VRAM": None,
        "OnPremise": False,
    },
    {
        "Category": "Mid-to-Large Open",
        "Model": "Gemma4-31B",
        "ST_ToolAcc": 94.55,
        "ST_StrictAST": 85.45,
        "MT_StrictAST": 64.17,
        "Macro_StrictAST": 74.81,
        "ArgMatch": 85.45,
        "MacroF1": 76.12,
        "FormatValid": 100.00,
        "Latency": 1.35,
        "Throughput": 42.50,
        "VRAM": 28.0,
        "OnPremise": True,
    },
    {
        "Category": "Mid-to-Large Open",
        "Model": "GPT-OSS-20B",
        "ST_ToolAcc": 89.09,
        "ST_StrictAST": 87.27,
        "MT_StrictAST": 57.10,
        "Macro_StrictAST": 72.19,
        "ArgMatch": 87.27,
        "MacroF1": 74.90,
        "FormatValid": 97.44,
        "Latency": 2.65,
        "Throughput": 22.80,
        "VRAM": 18.0,
        "OnPremise": True,
    },
    {
        "Category": "Mid-to-Large Open",
        "Model": "GPT-OSS-120B",
        "ST_ToolAcc": 85.45,
        "ST_StrictAST": 80.00,
        "MT_StrictAST": 59.82,
        "Macro_StrictAST": 69.91,
        "ArgMatch": 80.00,
        "MacroF1": 72.35,
        "FormatValid": 98.72,
        "Latency": 1.52,
        "Throughput": 38.20,
        "VRAM": 75.0,
        "OnPremise": True,
    },
    {
        "Category": "Mid-to-Large Open",
        "Model": "Nemotron-30B",
        "ST_ToolAcc": 61.82,
        "ST_StrictAST": 61.82,
        "MT_StrictAST": 14.24,
        "Macro_StrictAST": 38.03,
        "ArgMatch": 61.82,
        "MacroF1": 34.50,
        "FormatValid": 68.59,
        "Latency": 3.58,
        "Throughput": 18.40,
        "VRAM": 26.0,
        "OnPremise": True,
    },
    {
        "Category": "Proposed LoRA SLM",
        "Model": "TelkomNusa-LLM-7B",
        "ST_ToolAcc": 89.09,
        "ST_StrictAST": 83.64,
        "MT_StrictAST": 33.04,
        "Macro_StrictAST": 58.34,
        "ArgMatch": 83.64,
        "MacroF1": 74.20,
        "FormatValid": 100.00,
        "Latency": 11.02,
        "Throughput": 4.85,
        "VRAM": 9.2,
        "OnPremise": True,
    },
    {
        "Category": "Proposed LoRA SLM",
        "Model": "TelkomNusa-SLM-3B",
        "ST_ToolAcc": 85.45,
        "ST_StrictAST": 72.73,
        "MT_StrictAST": 16.38,
        "Macro_StrictAST": 44.55,
        "ArgMatch": 72.73,
        "MacroF1": 68.45,
        "FormatValid": 100.00,
        "Latency": 10.30,
        "Throughput": 5.12,
        "VRAM": 5.8,
        "OnPremise": True,
    },
    {
        "Category": "Open-Weight Baseline",
        "Model": "Llama-3.2-3B-Instruct",
        "ST_ToolAcc": 81.82,
        "ST_StrictAST": 78.18,
        "MT_StrictAST": 3.99,
        "Macro_StrictAST": 41.09,
        "ArgMatch": 78.18,
        "MacroF1": 42.10,
        "FormatValid": 100.00,
        "Latency": 22.03,
        "Throughput": 2.15,
        "VRAM": 6.1,
        "OnPremise": True,
    },
    {
        "Category": "Open-Weight Baseline",
        "Model": "Qwen2.5-7B-Instruct",
        "ST_ToolAcc": 85.45,
        "ST_StrictAST": 81.82,
        "MT_StrictAST": 0.00,
        "Macro_StrictAST": 40.91,
        "ArgMatch": 81.82,
        "MacroF1": 41.80,
        "FormatValid": 98.18,
        "Latency": 5.93,
        "Throughput": 8.58,
        "VRAM": 9.2,
        "OnPremise": True,
    },
    {
        "Category": "Open-Weight Baseline",
        "Model": "Qwen2.5-3B-Instruct",
        "ST_ToolAcc": 67.27,
        "ST_StrictAST": 61.82,
        "MT_StrictAST": 4.13,
        "Macro_StrictAST": 32.98,
        "ArgMatch": 61.82,
        "MacroF1": 47.26,
        "FormatValid": 99.01,
        "Latency": 7.65,
        "Throughput": 6.63,
        "VRAM": 5.8,
        "OnPremise": True,
    },
    {
        "Category": "Open-Weight Baseline",
        "Model": "Phi-4-mini",
        "ST_ToolAcc": 10.91,
        "ST_StrictAST": 10.91,
        "MT_StrictAST": 0.00,
        "Macro_StrictAST": 5.46,
        "ArgMatch": 10.91,
        "MacroF1": 4.80,
        "FormatValid": 97.03,
        "Latency": 6.68,
        "Throughput": 7.40,
        "VRAM": 6.8,
        "OnPremise": True,
    },
]

CATEGORY_COLORS = {
    "Proprietary Cloud": "#5C6BC0",
    "Mid-to-Large Open": "#1E88E5",
    "Proposed LoRA SLM": "#D32F2F",
    "Open-Weight Baseline": "#757575",
}


def plot_grouped_ast(df: pd.DataFrame, output_dir: str) -> None:
    """Plot single-turn, multi-turn, and macro-average Strict AST accuracy."""
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    x = np.arange(len(df))
    width = 0.26

    bars_st = ax.bar(
        x - width,
        df["ST_StrictAST"],
        width,
        label="Single-Turn Strict AST (%)",
        color="#1E88E5",
        edgecolor="#222222",
        linewidth=0.6,
    )
    bars_mt = ax.bar(
        x,
        df["MT_StrictAST"],
        width,
        label="Multi-Turn Strict AST (%)",
        color="#F57C00",
        edgecolor="#222222",
        linewidth=0.6,
    )
    bars_macro = ax.bar(
        x + width,
        df["Macro_StrictAST"],
        width,
        label="Macro-Average Strict AST (%)",
        color="#2E7D32",
        edgecolor="#222222",
        linewidth=0.6,
    )

    for bars in (bars_st, bars_mt, bars_macro):
        for bar in bars:
            h = bar.get_height()
            if h >= 5.0:
                ax.annotate(
                    f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    rotation=90,
                )

    ax.set_ylabel("Strict AST Execution Accuracy (%)")
    ax.set_title("Single-Turn vs. Multi-Turn BSS Execution Retention Across 12 Model Configurations", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=28, ha="right", fontsize=9)
    ax.set_ylim(0, 108)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_ast_retention_comparison.png"), dpi=300)
    plt.close(fig)


def plot_radar_profile(df: pd.DataFrame, output_dir: str) -> None:
    """Plot five-metric radar comparison for selected representative models."""
    metrics = ["ST_ToolAcc", "ArgMatch", "Macro_StrictAST", "MacroF1", "FormatValid"]
    labels = ["Tool Selection", "Arg Match", "Macro AST", "Macro F1", "Format Validity"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    selected = [
        ("TelkomNusa-LLM-7B", "#D32F2F", 2.4, 0.16),
        ("TelkomNusa-SLM-3B", "#F57C00", 2.2, 0.12),
        ("Gemma4-31B", "#1E88E5", 1.8, 0.05),
        ("Claude-Opus-5", "#5C6BC0", 1.8, 0.05),
        ("Qwen2.5-7B-Instruct", "#757575", 1.5, 0.04),
    ]

    fig, ax = plt.subplots(figsize=(7.8, 7.2), subplot_kw={"polar": True})
    for model_name, color, lw, alpha in selected:
        row = df[df["Model"] == model_name].iloc[0]
        values = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, values, color=color, linewidth=lw, label=model_name)
        ax.fill(angles, values, color=color, alpha=alpha)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color="#555555")
    ax.set_title("Five-Metric Performance Profile for Representative Models", pad=20)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_multidimensional_radar.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vram_efficiency_tradeoff(df: pd.DataFrame, output_dir: str) -> None:
    """Plot peak VRAM consumption versus Macro Strict AST for on-premise models."""
    on_prem = df[df["OnPremise"]].copy()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    # Trace non-dominated on-premise configurations (minimizing VRAM, maximizing Macro AST).
    sorted_df = on_prem.sort_values(by=["VRAM", "Macro_StrictAST"], ascending=[True, False])
    nondominated = []
    best_ast = -1.0
    for _, row in sorted_df.iterrows():
        if row["Macro_StrictAST"] > best_ast:
            nondominated.append((row["VRAM"], row["Macro_StrictAST"]))
            best_ast = row["Macro_StrictAST"]
    if len(nondominated) >= 2:
        px, py = zip(*nondominated)
        ax.plot(px, py, linestyle="--", color="#B71C1C", linewidth=1.3, alpha=0.75, label="Non-Dominated Envelope")

    offsets = {
        "TelkomNusa-SLM-3B": (0.8, 3.0),
        "Qwen2.5-3B-Instruct": (0.8, -4.5),
        "Llama-3.2-3B-Instruct": (1.2, -1.5),
        "Phi-4-mini": (1.2, 2.0),
        "TelkomNusa-LLM-7B": (1.2, 2.5),
        "Qwen2.5-7B-Instruct": (1.2, -3.5),
        "GPT-OSS-20B": (1.5, 2.5),
        "Nemotron-30B": (1.5, -3.0),
        "Gemma4-31B": (1.5, 2.5),
        "GPT-OSS-120B": (-12.0, -5.0),
    }

    for _, row in on_prem.iterrows():
        color = CATEGORY_COLORS[row["Category"]]
        is_proposed = row["Category"] == "Proposed LoRA SLM"
        marker = "*" if is_proposed else "o"
        size = 240 if is_proposed else 110
        ax.scatter(
            row["VRAM"],
            row["Macro_StrictAST"],
            s=size,
            color=color,
            marker=marker,
            edgecolor="#222222",
            linewidth=0.9,
            zorder=4 if is_proposed else 3,
        )
        dx, dy = offsets.get(row["Model"], (1.0, 2.0))
        ax.annotate(
            row["Model"],
            xy=(row["VRAM"], row["Macro_StrictAST"]),
            xytext=(row["VRAM"] + dx, row["Macro_StrictAST"] + dy),
            fontsize=8.5,
            fontweight="bold" if is_proposed else "normal",
        )

    ax.axvline(x=16.0, color="#666666", linestyle=":", linewidth=1.2, label="16 GB Single-GPU Threshold")
    ax.set_xlabel("Peak VRAM Consumption (GB)")
    ax.set_ylabel("Macro Strict AST Accuracy (%)")
    ax.set_title("On-Premise Accuracy vs. VRAM Trade-Off Across Evaluable Deployments", pad=12)
    ax.set_xlim(0, 82)
    ax.set_ylim(0, 86)
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for cat, color in CATEGORY_COLORS.items():
        if cat == "Proprietary Cloud":
            continue
        ax.scatter([], [], color=color, s=80, label=cat, edgecolor="#222222")
    ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", loc="lower right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_vram_efficiency_tradeoff.png"), dpi=300)
    plt.close(fig)


def main():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(BENCHMARK_DATA)
    plot_grouped_ast(df, output_dir)
    plot_radar_profile(df, output_dir)
    plot_vram_efficiency_tradeoff(df, output_dir)
    print(f"Generated figures in {output_dir}")


if __name__ == "__main__":
    main()
