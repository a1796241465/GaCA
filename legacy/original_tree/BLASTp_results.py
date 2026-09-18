import pandas as pd
import numpy as np



test_df = pd.read_csv('/home/lihaotian/new_ec/CARE_datasets/splits/task1/30_protein_test.csv')
test_df['Entry'] = test_df['Entry'].astype(str)


blast_df = pd.read_csv('full_BLASTp_results/full_blastp_results.txt', sep='\t',
                       names=['qseqid', 'sseqid', 'pident', 'evalue'])



blast_df['predicted_ec'] = blast_df['sseqid'].str.split('|').str[1]
blast_df['query_entry'] = blast_df['qseqid'].str.split('|').str[0]



blast_df = blast_df.sort_values('evalue').drop_duplicates('query_entry')




merged_df = pd.merge(test_df, blast_df, left_on='Entry', right_on='query_entry', how='left')



def check_accuracy(row):
    true_ec = str(row['EC number'])
    pred_ec = str(row['predicted_ec'])


    if pd.isna(row['predicted_ec']) or pred_ec == 'nan':
        return 0, 0, 0, 0

    true_parts = true_ec.split('.')
    pred_parts = pred_ec.split('.')

    acc = []

    for i in range(4):

        if i < len(true_parts) and i < len(pred_parts):

            if true_parts[:i + 1] == pred_parts[:i + 1]:
                acc.append(1)
            else:
                acc.append(0)
        else:
            acc.append(0)

    return acc



results = merged_df.apply(check_accuracy, axis=1)
results_df = pd.DataFrame(results.tolist(), columns=['L1', 'L2', 'L3', 'L4'])


print("=" * 30)
print(f"评估样本总数 (分母): {len(merged_df)}")
print(f"成功比对样本数: {len(blast_df)}")
print(f"未比对样本数 (自动判错): {len(merged_df) - len(blast_df)}")
print("-" * 30)
print(f"Level 1 Accuracy: {results_df['L1'].mean() * 100:.2f}%")
print(f"Level 2 Accuracy: {results_df['L2'].mean() * 100:.2f}%")
print(f"Level 3 Accuracy: {results_df['L3'].mean() * 100:.2f}%")
print(f"Level 4 Accuracy: {results_df['L4'].mean() * 100:.2f}%")
print("=" * 30)