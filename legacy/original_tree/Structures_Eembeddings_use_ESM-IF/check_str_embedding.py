import pickle
import numpy as np
import torch
import random
from scipy.spatial.distance import cosine



PKL_PATH = "train_structure_embeddings_esm_if.pkl"




def check_embedding(path):
    print(f"🧐 正在检查: {path} ...")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    ids = list(data.keys())
    print(f"✅ 成功加载，样本数: {len(ids)}")


    sample_id = ids[0]
    emb = data[sample_id]
    print(f"\n📊 [样本抽查] ID: {sample_id}")
    print(f"   - 形状: {emb.shape} (预期应为 (512,))")
    print(f"   - 类型: {emb.dtype}")
    print(f"   - 前10位数值: {emb[:10]}")
    print(f"   - 均值: {emb.mean():.4f} | 方差: {emb.var():.4f}")

    if emb.shape != (512,):
        print("❌ 警告: 维度不对！请检查是否做了 Mean Pooling。")


    all_embs = np.stack(list(data.values()))
    if np.isnan(all_embs).any():
        print("❌ 严重错误: 发现 NaN (非数字)！")
    elif np.isinf(all_embs).any():
        print("❌ 严重错误: 发现 Inf (无穷大)！")
    else:
        print("✅ 数值正常 (无 NaN/Inf)")


    zero_cnt = np.sum(np.all(all_embs == 0, axis=1))
    if zero_cnt > 0:
        print(f"⚠️ 警告: 发现 {zero_cnt} 个全 0 向量！可能是解析失败的样本。")
    else:
        print("✅ 无全 0 向量，所有样本均有特征。")



    id1, id2 = random.sample(ids, 2)
    sim = 1 - cosine(data[id1], data[id2])
    print(f"\n👯 [区分度检查]")
    print(f"   - 样本A ({id1}) vs 样本B ({id2})")
    print(f"   - 余弦相似度: {sim:.4f}")

    if sim > 0.9999:
        print("❌ 严重警告: 不同样本特征几乎完全一样！模型可能发生崩塌或输入有误。")
    else:
        print("✅ 特征具有区分度 (相似度不为 1)。")


if __name__ == "__main__":
    check_embedding(PKL_PATH)