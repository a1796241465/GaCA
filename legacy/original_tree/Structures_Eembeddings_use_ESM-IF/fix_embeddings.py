import pickle
import numpy as np
import os
from tqdm import tqdm



FILES_TO_FIX = [
    "train_structure_embeddings_esm_if.pkl",
    "test_structure_embeddings_esm_if.pkl"
]




def fix_pickle(path):
    print(f"🔧 正在修复: {path}")
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return

    with open(path, 'rb') as f:
        data = pickle.load(f)

    new_data = {}
    fixed_count = 0

    for pid, emb in tqdm(data.items()):

        if emb.shape == (1, 512):

            new_emb = emb.squeeze()
            new_data[pid] = new_emb
            fixed_count += 1
        else:

            new_data[pid] = emb


    with open(path, 'wb') as f:
        pickle.dump(new_data, f)

    print(f"✅ 修复完成！共处理 {len(data)} 条，修正了 {fixed_count} 条。")
    print(f"   现在形状统一为 (512,) 了。\n")


if __name__ == "__main__":
    print("🚀 开始批量修复维度问题...")
    for f in FILES_TO_FIX:
        fix_pickle(f)
    print("🎉 全部搞定！现在可以再次运行 check_quality.py 验证了。")