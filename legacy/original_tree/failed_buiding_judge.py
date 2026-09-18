import json

with open('dataset_stats.json') as f:
    stats = json.load(f)

print('=' * 60)
print('失败样本详情')
print('=' * 60)

failed = stats['failed']['validation_failed']
print(f'总共 {len(failed)} 个失败样本\n')

from collections import Counter
reasons = [item['error'] for item in failed]
reason_counts = Counter(reasons)

print('失败原因统计:')
for reason, count in reason_counts.most_common():
    print(f'  {reason}: {count} 个')

print('\n前20个失败样本:')
for i, item in enumerate(failed[:20], 1):
    print(f'{i:2d}. {item["id"]}: {item["error"]}')