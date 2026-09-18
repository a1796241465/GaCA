import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


sns.set(style="white", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'





def plot_hierarchical_heatmap():

    data = {
        'Method': ['RF', 'SVM', 'KNN', 'BLASTp', 'Lightgbm', 'ProtBERT', 'GACMA (Ours)'],
        'Level 1': [80.5, 82.8, 82.8, 89.1, 68.1, 87.9, 96.5],
        'Level 2': [73.4, 74.4, 75.2, 83.6, 56.4, 81.1, 91.8],
        'Level 3': [70.2, 71.5, 71.5, 79.2, 53.9, 79.0, 87.8],
        'Level 4': [64.6, 62.9, 63.5, 70.5, 48.6, 71.9, 81.6]
    }

    df = pd.DataFrame(data)


    df = df.sort_values(by='Level 4', ascending=False)
    df = df.set_index('Method')

    plt.figure(figsize=(9, 6))


    ax = sns.heatmap(df,
                     annot=True,
                     fmt=".1f",
                     cmap="YlGnBu",
                     linewidths=1,
                     linecolor='white',
                     cbar_kws={'label': 'Accuracy (%)'},
                     annot_kws={"weight": "bold"})


    plt.title("Hierarchical Accuracy Comparison (30-50% Identity)", fontweight='bold', fontsize=15, pad=20)
    plt.ylabel("Prediction Methods", fontweight='bold')
    plt.xlabel("Enzyme Commission (EC) Hierarchy Levels", fontweight='bold')


    plt.yticks(rotation=0)
    plt.xticks(fontweight='bold')

    plt.tight_layout()

    plt.savefig("Fig_Heatmap_Levels_30_50.pdf", dpi=300)
    print("✅ 30-50%热力图已生成: Fig_Heatmap_Levels_30_50.pdf")


if __name__ == "__main__":
    plot_hierarchical_heatmap()