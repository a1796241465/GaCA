import pickle
import pandas as pd


pkl_path = '/home/lihaotian/new_ec/processed_multimodal_datasets/multimodal_test_30.pkl'

print(f"正在读取: {pkl_path} ...")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("=" * 30)
print(f"数据总类型: {type(data)}")

if hasattr(data, '__len__'):
    print(f"数据长度: {len(data)}")


if isinstance(data, list) and len(data) > 0:
    first_item = data[0]
    print(f"列表第一个元素类型: {type(first_item)}")

    if isinstance(first_item, dict):
        print(f"【关键】该元素的 Keys: {list(first_item.keys())}")
    elif hasattr(first_item, '__dict__'):
        print(f"【关键】该元素的属性: {first_item.__dict__.keys()}")
    else:
        print(f"第一个元素内容: {first_item}")

elif isinstance(data, pd.DataFrame):
    print(f"【关键】DataFrame Columns: {list(data.columns)}")
elif isinstance(data, dict):
    print(f"【关键】字典 Keys (前5个): {list(data.keys())[:5]}")

print("=" * 30)