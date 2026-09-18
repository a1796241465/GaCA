import pickle
import numpy as np


file_path = '/home/lihaotian/new_ec/processed_multimodal_datasets/multimodal_dataset_train_50.pkl'

print(f"正在加载数据: {file_path} ...")
try:
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)

    print(f"数据加载成功！总样本数: {len(dataset)}")


    sample = dataset[0]
    print("-" * 30)
    print("【样本 0 结构详情】")
    print(f"数据类型: {type(sample)}")

    if isinstance(sample, dict):
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"Key: ['{key}'] \t| Shape: {value.shape} \t| Type: {type(value)}")
            elif isinstance(value, str):

                print(f"Key: ['{key}'] \t| Value: {value[:20]}... (Len: {len(value)})")
            else:
                print(f"Key: ['{key}'] \t| Value: {value}")
    else:

        print("样本不是字典，尝试打印属性:", dir(sample))

except Exception as e:
    print(f"读取出错: {e}")