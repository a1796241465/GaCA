import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('train_final_with_structure.csv')
details = pd.read_csv('download_details.csv')

print("=" * 60)
print("📊 最终数据集质量报告")
print("=" * 60)


print(f"\n【基本信息】")
print(f"总样本数: {len(df)}")
print(f"EC编号数: {df['EC number'].nunique()}")
print(f"平均序列长度: {df['Length'].mean():.1f} aa")
print(f"序列长度范围: {df['Length'].min()}-{df['Length'].max()} aa")


print(f"\n【EC层级分布】")
print(f"EC Level 1: {df['EC1'].nunique()} 类")
print(f"EC Level 2: {df['EC2'].nunique()} 类")
print(f"EC Level 3: {df['EC3'].nunique()} 类")
print(f"EC Level 4: {df['EC number'].nunique()} 类")


success_details = details[details['status'] == 'downloaded']
if not success_details.empty and 'confidence' in success_details.columns:
    print(f"\n【结构质量】")
    print(f"平均置信度: {success_details['confidence'].mean():.1f}")
    print(f"高置信度(>90): {(success_details['confidence'] > 90).sum()} 个")
    print(f"中等置信度(70-90): {((success_details['confidence'] >= 70) & (success_details['confidence'] <= 90)).sum()} 个")
    print(f"低置信度(<70): {(success_details['confidence'] < 70).sum()} 个")


ec_counts = df['EC number'].value_counts()
print(f"\n【样本分布】")
print(f"样本数最多的EC: {ec_counts.iloc[0]} 个")
print(f"样本数最少的EC: {ec_counts.iloc[-1]} 个")
print(f"平均每个EC: {ec_counts.mean():.1f} 个")
print(f"中位数: {ec_counts.median():.0f} 个")

print("\n✅ 数据集准备完成！可以开始模型训练了！")