import torch
import esm
import pickle
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")



TRAIN_PKL_PATH = '/home/lihaotian/new_ec/processed_multimodal_datasets/multimodal_dataset_train_50.pkl'
TEST_30_PKL_PATH = '/home/lihaotian/new_ec/processed_multimodal_datasets/multimodal_test_30.pkl'
TEST_30_50_CSV_PATH = '/home/lihaotian/new_ec/test_30_50_clean.csv'


OUTPUT_DIR = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/'
TRAIN_OUT_NAME = '1024_train_seq_embeddings_esm2.pkl'
TEST_30_OUT_NAME = '1024_test_seq_embeddings_esm2.pkl'
TEST_30_50_OUT_NAME = '30_50_seq_embeddings_esm2.pkl'


MODEL_NAME = "esm2_t33_650M_UR50D"
BATCH_SIZE = 16
MAX_LEN = 1024




class ProteinSeqDatasetPKL(Dataset):
    """处理 Pickle 格式的数据集"""

    def __init__(self, pkl_path, key_id='protein_id', key_seq='sequence_raw'):
        print(f"📖 正在加载 PKL 数据集: {pkl_path} ...")
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.key_id = key_id
        self.key_seq = key_seq
        print(f"   ✅ 加载了 {len(self.data)} 条数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item.get(self.key_id, f"Unknown_{idx}"), item.get(self.key_seq, "")


class ProteinSeqDatasetCSV(Dataset):
    """处理 CSV 格式的数据集"""

    def __init__(self, csv_path):
        print(f"📖 正在加载 CSV 数据集: {csv_path} ...")
        self.df = pd.read_csv(csv_path)
        if 'Entry' not in self.df.columns or 'Sequence' not in self.df.columns:
            raise ValueError(f"CSV 必须包含 'Entry' 和 'Sequence' 列")
        print(f"   ✅ 加载了 {len(self.df)} 条数据")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return str(row['Entry']), str(row['Sequence'])




def extract_embeddings(dataset, output_path, model, batch_converter, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
    embeddings = {}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"🚀 开始提取特征 (Max Len = {MAX_LEN}, 【已移除Mean Pooling，保留残基级二维矩阵，Float16压缩】)...")

    with torch.no_grad():
        for batch in tqdm(loader):
            ids, seqs = zip(*batch)


            truncated_seqs = [s[:MAX_LEN] for s in seqs]


            labels, strs, tokens = batch_converter([(i, s) for i, s in zip(ids, truncated_seqs)])
            tokens = tokens.to(device)


            results = model(tokens, repr_layers=[33], return_contacts=False)
            token_reps = results["representations"][33]


            for i, protein_id in enumerate(ids):
                seq_len = len(strs[i])

                seq_emb = token_reps[i, 1: seq_len + 1]


                raw_emb = seq_emb.cpu().numpy().astype(np.float16)
                embeddings[protein_id] = raw_emb

    print(f"💾 保存 {len(embeddings)} 条特征到 -> {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print("✅ 完成!\n")




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 运行设备: {device}")


    print(f"📥 正在加载 ESM-2 模型 ({MODEL_NAME})...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()


    if os.path.exists(TRAIN_PKL_PATH):
        print("=" * 50)
        print("🔄 任务 1/3: 处理训练集 (PKL)...")
        ds_train = ProteinSeqDatasetPKL(TRAIN_PKL_PATH)
        extract_embeddings(ds_train, os.path.join(OUTPUT_DIR, TRAIN_OUT_NAME), model, batch_converter, device)


    if os.path.exists(TEST_30_PKL_PATH):
        print("=" * 50)
        print("🔄 任务 2/3: 处理 <30% 测试集 (PKL)...")
        ds_test_30 = ProteinSeqDatasetPKL(TEST_30_PKL_PATH)
        extract_embeddings(ds_test_30, os.path.join(OUTPUT_DIR, TEST_30_OUT_NAME), model, batch_converter, device)


    if os.path.exists(TEST_30_50_CSV_PATH):
        print("=" * 50)
        print("🔄 任务 3/3: 处理 30-50% 测试集 (CSV)...")
        ds_test_30_50 = ProteinSeqDatasetCSV(TEST_30_50_CSV_PATH)
        extract_embeddings(ds_test_30_50, os.path.join(OUTPUT_DIR, TEST_30_50_OUT_NAME), model, batch_converter, device)

    print("🎉 全部序列特征提取任务圆满完成！")


if __name__ == "__main__":
    main()