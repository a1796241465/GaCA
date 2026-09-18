import torch
import esm
import pickle
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



TEST_CSV_PATH = '/home/lihaotian/new_ec/test_30_50_clean.csv'


OUTPUT_DIR = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/'
TEST_OUT_NAME = '30_50_seq_embeddings_esm2_mean.pkl'


MODEL_NAME = "esm2_t33_650M_UR50D"
BATCH_SIZE = 16
MAX_LEN = 1024




class ProteinSeqDataset(Dataset):
    def __init__(self, csv_path):
        print(f"📖 正在加载 CSV 数据集: {csv_path} ...")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 文件不存在: {csv_path}")

        self.df = pd.read_csv(csv_path)


        if 'Entry' not in self.df.columns or 'Sequence' not in self.df.columns:
            raise ValueError(f"CSV 必须包含 'Entry' 和 'Sequence' 列")

        print(f"   ✅ 加载了 {len(self.df)} 条数据")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row['Entry'])
        seq = str(row['Sequence'])
        return pid, seq


def extract_embeddings(data_path, output_path, model, batch_converter, device):
    if not os.path.exists(data_path):
        print(f"⚠️ 跳过 {data_path} (文件不存在)")
        return

    dataset = ProteinSeqDataset(data_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x, num_workers=4)

    embeddings = {}


    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"🚀 开始提取特征 (Max Len = {MAX_LEN}, Mean Pooling)...")

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
                mean_emb = seq_emb.mean(dim=0).cpu().numpy()

                embeddings[protein_id] = mean_emb


    print(f"💾 保存 {len(embeddings)} 条特征到 -> {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print("✅ 完成!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 运行设备: {device}")


    print(f"📥 正在加载 ESM-2 模型 ({MODEL_NAME})...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()


    output_pkl_path = os.path.join(OUTPUT_DIR, TEST_OUT_NAME)
    print(f"\n🔄 处理测试集: {TEST_CSV_PATH} -> {output_pkl_path}")
    extract_embeddings(TEST_CSV_PATH, output_pkl_path, model, batch_converter, device)


if __name__ == "__main__":
    main()