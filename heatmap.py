import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_move_heatmap(ax, df, id_agent):
    """Plots a heatmap of move coverage for a specific agent."""
    if not {"dest_x", "dest_y", "agent"}.issubset(df.columns):
        print("Error: columns dest_x, dest_y and agent missing.")
        return
    grid_size = 5
    heatmap_data = np.zeros((grid_size, grid_size))
    df_agent = df[df["agent"] == id_agent]
    for _, row in df_agent.iterrows():
        x, y = int(row["dest_x"]), int(row["dest_y"])
        if 0 <= x < grid_size and 0 <= y < grid_size:
            heatmap_data[x, y] += 1
    color = "Blues" if id_agent == 0 else "Reds"
    title = f"Heatmap agent {id_agent} ({'Blue' if id_agent == 0 else 'Red'})"
    sns.heatmap(heatmap_data, cmap=color, annot=True, fmt=".0f",
                linewidths=0.5, linecolor="black", cbar=True, ax=ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
    path = sys.argv[1]
    df = pd.read_csv(path)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    plot_move_heatmap(axes[0], df, 0)
    plot_move_heatmap(axes[1], df, 1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
