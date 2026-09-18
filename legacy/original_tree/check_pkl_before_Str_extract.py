import pickle
import numpy as np
import torch


file_path = 'multimodal_dataset_clean.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f)

sample = data[0]
coords = sample['structure_coords']
seq_len = len(sample['sequence_raw'])

print(f"样本 ID: {sample['protein_id']}")
print(f"序列长度: {seq_len}")
print(f"坐标形状: {coords.shape}")
print(f"坐标数据类型: {coords.dtype}")
print(f"前5个坐标:\n{coords[:5]}")


if np.isnan(coords).any():
    print("警告！坐标中包含 NaN！")
else:
    print("坐标数据正常，无 NaN。")