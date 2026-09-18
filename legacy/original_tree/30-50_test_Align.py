import pandas as pd
import os



ORIGINAL_CSV = '/home/lihaotian/new_ec/CARE_datasets/splits/task1/30-50_protein_test.csv'

STRUCTURE_DIR = 'test_30-50_structures'

OUTPUT_CSV = 'test_30_50_clean.csv'


def run():
    print(f"Loading {ORIGINAL_CSV}...")
    df = pd.read_csv(ORIGINAL_CSV)

    valid_indices = []
    missing_count = 0

    for idx, row in df.iterrows():
        uid = str(row['Entry'])
        pdb_path = os.path.join(STRUCTURE_DIR, f"{uid}.pdb")

        if os.path.exists(pdb_path):
            valid_indices.append(idx)
        else:
            missing_count += 1

    df_valid = df.loc[valid_indices].copy()
    df_valid.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 40)
    print(f"📊 对齐报告")
    print(f"   原始样本数: {len(df)}")
    print(f"   结构文件数: {len(df_valid)}")
    print(f"   丢失样本数: {missing_count}")
    print(f"✅ 已保存清洗后的表格: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    run()