import pickle
import numpy as np
import torch


STR_TEST_PKL = '/home/lihaotian/new_ec/Training_structures_only/v1_structure_embeddings_residue_test.pkl'
RAW_TEST_PKL = '/home/lihaotian/new_ec/processed_multimodal_datasets/multimodal_test_30.pkl'


def inspect_data():
    print(f"正在检查文件: {STR_TEST_PKL} ...")

    with open(STR_TEST_PKL, 'rb') as f:
        str_data = pickle.load(f)

    print(f"1. 结构数据类型: {type(str_data)}")
    print(f"2. 结构数据长度: {len(str_data)}")

    if len(str_data) > 0:
        first_item = str_data[0]
        print(f"3. 第一条数据的类型: {type(first_item)}")
        print("-" * 30)
        print("【第一条数据内容预览】:")
        print(first_item)
        print("-" * 30)


        if isinstance(first_item, dict) and ('protein_id' in first_item or 'id' in first_item):
            print("✅ 结果判定：你的数据包含 ID，可以直接运行清洗脚本！")
            if 'protein_id' in first_item:
                print("   (ID 键名为 'protein_id')")
            else:
                print("   (ID 键名为 'id'，请在清洗脚本中对应修改)")
        else:
            print("❌ 结果判定：你的数据是纯数组（不含 ID），无法直接清洗！")
            print("   原因：无法知道这第 1 条特征属于 Raw 数据里的哪一个蛋白质。")


    with open(RAW_TEST_PKL, 'rb') as f:
        raw_data = pickle.load(f)
        print(f"\n参照 - Raw 数据第一条 ID: {raw_data[0].get('protein_id', 'No ID found')}")


if __name__ == "__main__":
    inspect_data()