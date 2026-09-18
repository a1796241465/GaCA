import pandas as pd
import pickle
import os



GACMA_TEST_PKL = '/home/lihaotian/new_ec/processed_multimodal_datasets/multimodal_test_30.pkl'



TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'


RAW_TEST_CSV = '/home/lihaotian/new_ec/CARE_datasets/splits/task1/30_protein_test.csv'


BLAST_RESULT = '/home/lihaotian/new_ec/BLASTp_results/blastp_results.txt'



def main():
    print(">>> 正在进行最终公平评测 (Full Metrics)...")


    if not os.path.exists(GACMA_TEST_PKL):
        print(f"❌ 错误: 找不到 {GACMA_TEST_PKL}")
        return
    with open(GACMA_TEST_PKL, 'rb') as f:
        pkl_data = pickle.load(f)
    structure_ids = set([item['protein_id'] for item in pkl_data])


    if not os.path.exists(TRAIN_CSV):
        print(f"❌ 错误: 找不到 {TRAIN_CSV}")
        return
    train_df = pd.read_csv(TRAIN_CSV)
    seen_ecs = set(train_df['EC number'].unique())


    test_df = pd.read_csv(RAW_TEST_CSV)
    test_df['Entry'] = test_df['Entry'].astype(str)

    final_df = test_df[
        (test_df['Entry'].isin(structure_ids)) &
        (test_df['EC number'].isin(seen_ecs))
        ].copy()

    print(f"最终评测样本数 (分母): {len(final_df)}")


    blast_df = pd.read_csv(BLAST_RESULT, sep='\t', names=['qseqid', 'sseqid', 'pident', 'evalue'])
    blast_df['query_entry'] = blast_df['qseqid'].str.split('|').str[0]
    blast_df['predicted_ec'] = blast_df['sseqid'].str.split('|').str[1]

    blast_df = blast_df.sort_values('evalue').drop_duplicates('query_entry')
    merged_df = pd.merge(final_df, blast_df, left_on='Entry', right_on='query_entry', how='left')


    results = merged_df.apply(check_accuracy, axis=1)
    results_df = pd.DataFrame(results.tolist(), columns=['L1', 'L2', 'L3', 'L4'])

    print("\n" + "=" * 40)
    print(f"【BLASTp Final Baseline Results】 (N={len(merged_df)})")
    print("=" * 40)
    print(f"Level 1 Accuracy: {results_df['L1'].mean() * 100:.2f}%")
    print(f"Level 2 Accuracy: {results_df['L2'].mean() * 100:.2f}%")
    print(f"Level 3 Accuracy: {results_df['L3'].mean() * 100:.2f}%")
    print(f"Level 4 Accuracy: {results_df['L4'].mean() * 100:.2f}%")
    print("=" * 40)


def check_accuracy(row):
    true_ec = str(row['EC number'])
    if pd.isna(row['predicted_ec']): return 0, 0, 0, 0

    pred_ec = str(row['predicted_ec'])
    true_parts = true_ec.split('.')
    pred_parts = pred_ec.split('.')

    acc = []
    for i in range(4):
        if i < len(true_parts) and i < len(pred_parts):
            acc.append(1 if true_parts[:i + 1] == pred_parts[:i + 1] else 0)
        else:
            acc.append(0)
    return acc


if __name__ == "__main__":
    main()