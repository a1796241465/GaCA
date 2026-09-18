import sys
sys.path.insert(0, '/home/lihaotian/new_ec/CLEAN/CLEAN-main/app')

import pandas as pd
from CLEAN.infer import infer_maxsep


print("=" * 50)
print(">>> 正在推理 <30% 测试集...")
infer_maxsep("split70", "your_test30_tab", report_metrics=True, pretrained=True)

print("=" * 50)
print(">>> 正在推理 30-50% 测试集...")
infer_maxsep("split70", "your_test3050_tab", report_metrics=True, pretrained=True)



def hierarchical_accuracy(result_csv, gt_csv, dataset_name):
    print("\n" + "=" * 50)
    print(f">>> Hierarchical Accuracy: {dataset_name}")

    results = {}
    with open(result_csv) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            entry = parts[0]
            ec_pred = parts[1].replace('EC:', '').split('/')[0]
            results[entry] = ec_pred

    gt_df = pd.read_csv(gt_csv)
    c1 = c2 = c3 = c4 = total = 0
    not_found = 0

    for _, row in gt_df.iterrows():
        entry = str(row['Entry'])
        true_ec = str(row['EC number'])
        pred_ec = results.get(entry, '')

        if pred_ec == '':
            not_found += 1

        tp = true_ec.split('.')
        pp = pred_ec.split('.')

        if len(pp) >= 1 and tp[0] == pp[0]: c1 += 1
        if len(pp) >= 2 and tp[:2] == pp[:2]: c2 += 1
        if len(pp) >= 3 and tp[:3] == pp[:3]: c3 += 1
        if true_ec == pred_ec: c4 += 1
        total += 1

    print(f"   总样本数: {total} | 未预测到: {not_found}")
    print(f"   Level 1: {c1/total*100:.2f}%")
    print(f"   Level 2: {c2/total*100:.2f}%")
    print(f"   Level 3: {c3/total*100:.2f}%")
    print(f"   Level 4: {c4/total*100:.2f}%")
    print("=" * 50)


hierarchical_accuracy(
    "results/your_test30_tab_maxsep.csv",
    "data/your_test30.csv",
    "Test Set (<30% Seq. Identity)"
)

hierarchical_accuracy(
    "results/your_test3050_tab_maxsep.csv",
    "data/your_test3050.csv",
    "Test Set (30-50% Seq. Identity)"
)