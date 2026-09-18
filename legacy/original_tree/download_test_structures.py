
from download_train_structures import batch_download_alphafold_structures
import pandas as pd


df_test = pd.read_csv('/home/lihaotian/new_ec/CARE_datasets/splits/task1/30-50_protein_test.csv')


results, details = batch_download_alphafold_structures(
    df_test,
    save_dir='test_30-50_structures'
)

print(f"✅ 测试集结构下载完成: {len(results['success'])} 个")