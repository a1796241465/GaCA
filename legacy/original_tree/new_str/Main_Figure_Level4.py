import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


sns.set(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'




def plot_main_results():

    data = {
        'Dataset': [
            '<30% Identity', '<30% Identity', '<30% Identity', '<30% Identity',
            '<30% Identity', '<30% Identity', '<30% Identity',
            '30-50% Identity', '30-50% Identity', '30-50% Identity', '30-50% Identity',
            '30-50% Identity', '30-50% Identity', '30-50% Identity'
        ],
        'Method': [
            'BLASTp', 'KNN', 'SVM', 'RF', 'Lightgbm', 'ProtBERT', 'GACMA (Ours)',
            'BLASTp', 'KNN', 'SVM', 'RF', 'Lightgbm', 'ProtBERT', 'GACMA (Ours)'
        ],
        'Accuracy': [
            66.67, 38.68, 44.86, 27.16, 21.40, 45.05, 67.90,
            70.49, 63.52, 62.89, 64.57, 48.64, 71.91, 81.55
        ]
    }


    df = pd.DataFrame(data)

    plt.figure(figsize=(11, 6.5))


    custom_palette = [
        "#95a5a6", "#7f8c8d", "#34495e", "#5d6d7e",
        "#3498db", "#2980b9", "#e74c3c"
    ]


    ax = sns.barplot(x="Dataset", y="Accuracy", hue="Method", data=df, palette=custom_palette)


    plt.ylim(0, 100)
    plt.ylabel("Level 4 Accuracy (%)", fontweight='bold')
    plt.xlabel("", fontweight='bold')
    plt.title("Performance Comparison on CARE Benchmarks", fontweight='bold', fontsize=16)


    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)


    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2., height + 1,
                    '{:.1f}'.format(height),
                    ha="center", fontsize=8, color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig("Fig2_Main_Results.pdf", dpi=300)
    print("✅ Figure 2 生成完毕 (Fig2_Main_Results.pdf)")

if __name__ == "__main__":
    plot_main_results()