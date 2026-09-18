import torch
import esm
import pickle
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



TRAIN_DATA_PATH = '/home/lihaotian/new_ec/processed_multimodal_datasets/multimodal_dataset_train_50.pkl'
TEST_DATA_PATH = '/home/lihaotian/new_ec/processed_multimodal_datasets/multimodal_test_30.pkl'


OUTPUT_DIR = '/home/lihaotian/new_ec/Sequences_Embedding_use_ESM-2/'
TRAIN_OUT_NAME = '1024_train_seq_embeddings_esm2_mean.pkl'
TEST_OUT_NAME = '1024_test_seq_embeddings_esm2_mean.pkl'


MODEL_NAME = "esm2_t33_650M_UR50D"
BATCH_SIZE = 16
MAX_LEN = 1024


KEY_ID = 'protein_id'
KEY_SEQ = 'sequence_raw'




class ProteinSeqDataset(Dataset):
    def __init__(self, pkl_path):
        print(f"📖 正在加载数据集: {pkl_path} ...")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"❌ 文件不存在: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"   ✅ 加载了 {len(self.data)} 条数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pid = item.get(KEY_ID, f"Unknown_{idx}")
        seq = item.get(KEY_SEQ, "")
        return pid, seq


def extract_embeddings(data_path, output_path, model, batch_converter, device):
    if not os.path.exists(data_path):
        print(f"⚠️ 跳过 {data_path} (文件不存在)")
        return

    dataset = ProteinSeqDataset(data_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)

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


    print("\n" + "=" * 40)
    print("🔄 处理训练集...")
    extract_embeddings(TRAIN_DATA_PATH, os.path.join(OUTPUT_DIR, TRAIN_OUT_NAME), model, batch_converter, device)


    print("\n" + "=" * 40)
    print("🔄 处理测试集...")
    extract_embeddings(TEST_DATA_PATH, os.path.join(OUTPUT_DIR, TEST_OUT_NAME), model, batch_converter, device)

    print("\n🎉 全部任务完成！")


if __name__ == "__main__":
    main()