import pandas as pd


details = pd.read_csv('download_details.csv')

print("=" * 60)
print("📊 数据质量分析")
print("=" * 60)


total = len(details)
high_conf = len(details[details['confidence'] > 90])
mid_conf = len(details[(details['confidence'] >= 70) & (details['confidence'] <= 90)])
low_conf = len(details[details['confidence'] < 70])

print(f"总样本数: {total}")
print(f"\n质量分布:")
print(f"  高质量 (>90):  {high_conf:5d} ({high_conf/total*100:.1f}%)")
print(f"  中等质量(70-90): {mid_conf:5d} ({mid_conf/total*100:.1f}%)")
print(f"  低质量 (<70):   {low_conf:5d} ({low_conf/total*100:.1f}%)")


print("\n" + "=" * 60)
print("⚠️ 低质量样本详情")
print("=" * 60)

low_conf_samples = details[details['confidence'] < 70].sort_values('confidence')
print(low_conf_samples[['uniprot_id', 'confidence', 'version']].to_string())


print("\n" + "=" * 60)
print("💡 建议")
print("=" * 60)
print("推荐阈值: confidence >= 70")
print(f"  → 保留: {total - low_conf} 个样本 ({(total-low_conf)/total*100:.1f}%)")
print(f"  → 移除: {low_conf} 个样本 ({low_conf/total*100:.1f}%)")