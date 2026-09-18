import pandas as pd
import requests


df = pd.read_csv('train_filtered_final.csv')

print("=" * 60)
print("🔍 Entry列格式诊断")
print("=" * 60)


print("\n前10个Entry示例:")
for i, entry in enumerate(df['Entry'].head(10)):
    print(f"{i + 1}. {entry} (长度: {len(entry)}, 类型: {type(entry)})")


print("\n" + "=" * 60)
print("🌐 测试AlphaFold URL访问")
print("=" * 60)

test_entries = df['Entry'].head(5).tolist()

for entry in test_entries:

    urls = [
        f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F1-model_v4.pdb",
        f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F1-model_v3.pdb",
        f"https://alphafold.ebi.ac.uk/api/prediction/{entry}",
    ]

    print(f"\n测试 Entry: {entry}")
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            print(f"  URL: {url}")
            print(f"  状态码: {response.status_code}")
            if response.status_code == 200:
                print(f"  ✅ 成功!")
                break
            elif response.status_code == 404:
                print(f"  ❌ 404 Not Found")
            else:
                print(f"  ⚠️ 其他错误")
        except Exception as e:
            print(f"  🔥 异常: {e}")


print("\n" + "=" * 60)
print("🔍 Entry列特殊字符检查")
print("=" * 60)


special_chars = df['Entry'].apply(lambda x: any(c in str(x) for c in [' ', '\n', '\t', '\r']))
print(f"含有特殊字符的Entry数量: {special_chars.sum()}")


print(f"\nEntry长度统计:")
print(f"  最短: {df['Entry'].str.len().min()}")
print(f"  最长: {df['Entry'].str.len().max()}")
print(f"  平均: {df['Entry'].str.len().mean():.1f}")


print(f"\nEntry格式示例 (前20个):")
for entry in df['Entry'].head(20):
    print(f"  {entry}")