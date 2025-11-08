import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_results(df: pd.DataFrame, output_dir: Path):
    """
    Generates and saves comparison plots for the experiment results.
    """
    if df.empty:
        print("No data to plot.")
        return
        
    # Ensure 'bitrate_kbs' is numeric for plotting
    df['bitrate_kbs'] = pd.to_numeric(df['bitrate_kbs'])
    
    sns.set_theme(style="whitegrid")
    
    # Metrics to plot as line graphs (higher is better)
    line_metrics_higher_better = ["PSNR", "SSIM", "RecoveryProbability"]
    # Metrics to plot as line graphs (lower is better)
    line_metrics_lower_better = ["BER", "MSE", "BRISQUE"]

    codecs = df['codec'].unique()
    
    for codec in codecs:
        codec_df = df[df['codec'] == codec].copy()
        
        # --- Plot: Higher is Better ---
        for metric in line_metrics_higher_better:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=codec_df,
                x="bitrate_kbs",
                y=metric,
                hue="method",
                style="method",
                markers=True,
                dashes=False
            )
            plt.title(f"{metric} vs. Bitrate (Codec: {codec})")
            plt.xlabel("Bitrate (kbps)")
            plt.ylabel(metric)
            plt.legend(title="Method")
            plt.savefig(output_dir / f"plot_{codec}_{metric}_vs_bitrate.png")
            plt.close()

        # --- Plot: Lower is Better ---
        for metric in line_metrics_lower_better:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=codec_df,
                x="bitrate_kbs",
                y=metric,
                hue="method",
                style="method",
                markers=True,
                dashes=False
            )
            plt.title(f"{metric} vs. Bitrate (Codec: {codec})")
            plt.xlabel("Bitrate (kbps)")
            plt.ylabel(metric)
            plt.legend(title="Method")
            plt.savefig(output_dir / f"plot_{codec}_{metric}_vs_bitrate.png")
            plt.close()

    print(f"Generated plots saved to {output_dir}")