



















































import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


sns.set(style="white", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'





def plot_hierarchical_heatmap():

    data = {
        'Method': ['RF', 'SVM', 'KNN', 'BLASTp', 'Lightgbm', 'ProtBERT', 'GACMA (Ours)'],
        'Level 1': [51.4, 74.9, 64.2, 86.0, 43.2, 69.4, 88.1],
        'Level 2': [39.5, 63.4, 54.7, 80.6, 31.3, 62.2, 80.1],
        'Level 3': [35.4, 58.0, 49.4, 77.0, 27.2, 56.5, 77.8],
        'Level 4': [27.2, 44.9, 38.7, 66.7, 21.4, 45.1, 67.9]
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


    plt.title("Hierarchical Accuracy Comparison (<30% Identity)", fontweight='bold', fontsize=15, pad=20)
    plt.ylabel("Prediction Methods", fontweight='bold')
    plt.xlabel("Enzyme Commission (EC) Hierarchy Levels", fontweight='bold')


    plt.yticks(rotation=0)
    plt.xticks(fontweight='bold')

    plt.tight_layout()
    plt.savefig("Fig_Heatmap_Levels.pdf", dpi=300)
    print("✅ 多层级热力图已生成: Fig_Heatmap_Levels.pdf")


if __name__ == "__main__":
    plot_hierarchical_heatmap()