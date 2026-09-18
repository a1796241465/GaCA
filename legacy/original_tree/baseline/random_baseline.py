import numpy as np
import pandas as pd


TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'
TEST_30_CSV = '/home/lihaotian/new_ec/test_30_cleaned_with_structure.csv'
TEST_30_50_CSV = '/home/lihaotian/new_ec/test_30_50_clean.csv'

RANDOM_SEED = 42



def calc_hierarchical_metrics(y_true, y_pred):
    """计算分层准确率(EC号的4个层级)"""
    c1 = c2 = c3 = c4 = 0
    total = len(y_true)
    for t, p in zip(y_true, y_pred):
        tp, pp = t.split('.'), p.split('.')
        if tp[0] == pp[0]:
            c1 += 1
        if len(tp) >= 2 and len(pp) >= 2 and tp[:2] == pp[:2]:
            c2 += 1
        if len(tp) >= 3 and len(pp) >= 3 and tp[:3] == pp[:3]:
            c3 += 1
        if t == p:
            c4 += 1
    return c1 / total, c2 / total, c3 / total, c4 / total


def random_baseline(train_csv, test_csv, test_name):
    """
    Random baseline: 从训练集EC类别中随机选择
    
    Args:
        train_csv: 训练集CSV路径
        test_csv: 测试集CSV路径
        test_name: 测试集名称(用于打印)
    """
    print(f"\n{'='*60}")
    print(f"🎲 Random Baseline - {test_name}")
    print(f"{'='*60}")
    

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    train_labels = train_df['EC number'].astype(str).values
    test_labels = test_df['EC number'].astype(str).values
    
    print(f"Train samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Unique EC in train: {len(set(train_labels))}")
    

    np.random.seed(RANDOM_SEED)
    random_preds = np.random.choice(train_labels, size=len(test_labels), replace=True)
    

    l1, l2, l3, l4 = calc_hierarchical_metrics(test_labels, random_preds)
    

    print(f"\nResults:")
    print(f"  L1 (1st digit): {l1*100:.2f}%")
    print(f"  L2 (1-2 digit): {l2*100:.2f}%")
    print(f"  L3 (1-3 digit): {l3*100:.2f}%")
    print(f"  L4 (exact):     {l4*100:.2f}%")
    
    return {
        'test_name': test_name,
        'L1': l1,
        'L2': l2,
        'L3': l3,
        'L4': l4
    }



def main():
    print("Starting Random Baseline Evaluation...")
    
    results = []
    

    result1 = random_baseline(TRAIN_CSV, TEST_30_CSV, "Test_30")
    results.append(result1)
    

    result2 = random_baseline(TRAIN_CSV, TEST_30_50_CSV, "Test_30-50")
    results.append(result2)
    

    print(f"\n{'='*60}")
    print("📊 Random Baseline Summary")
    print(f"{'='*60}")
    print(f"{'Test Set':<15} | {'L1':<8} | {'L2':<8} | {'L3':<8} | {'L4':<8}")
    print("-" * 60)
    
    for res in results:
        print(f"{res['test_name']:<15} | {res['L1']*100:>6.2f}% | "
              f"{res['L2']*100:>6.2f}% | {res['L3']*100:>6.2f}% | {res['L4']*100:>6.2f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
