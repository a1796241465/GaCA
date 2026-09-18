"""
测试集多模态数据集构建脚本
与训练集使用完全相同的处理流程，确保数据一致性
"""

import sys
import argparse


from build_multimodal_dataset import (
    extract_structure_with_validation,
    align_sequences,
    encode_sequence,
    validate_sample,
    build_multimodal_dataset,
    test_dataset
)


def main():
    """构建测试集的主函数"""

    parser = argparse.ArgumentParser(description='构建测试集多模态数据集')
    parser.add_argument('--csv', type=str,
                        default='test_30_cleaned_with_structure.csv',
                        help='测试集CSV文件（清理后）')
    parser.add_argument('--structures', type=str,
                        default='test_structures',
                        help='测试集PDB文件目录')
    parser.add_argument('--output', type=str,
                        default='test_30_multimodal.pkl',
                        help='输出pkl文件名')
    parser.add_argument('--min-identity', type=float,
                        default=0.95,
                        help='最小序列一致性阈值')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🧪 开始构建测试集多模态数据集")
    print("=" * 70)
    print(f"输入CSV: {args.csv}")
    print(f"结构目录: {args.structures}")
    print(f"输出文件: {args.output}")
    print(f"序列一致性阈值: {args.min_identity}")
    print("=" * 70)


    dataset, stats = build_multimodal_dataset(
        csv_file=args.csv,
        structures_dir=args.structures,
        output_file=args.output,
        min_identity=args.min_identity
    )


    test_dataset(args.output)


    import json
    stats_file = args.output.replace('.pkl', '_stats.json')


    stats_serializable = {
        'dataset_type': 'test',
        'test_split': '30% sequence identity',
        'total': stats['total'],
        'success': stats['success'],
        'success_rate': f"{stats['success'] / stats['total'] * 100:.1f}%",
        'failed_counts': {
            k: len(v) for k, v in stats['failed'].items()
        },
        'warnings': {
            'length_adjusted': len(stats['warnings']['length_adjusted'])
        }
    }

    with open(stats_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ 测试集构建完成！")
    print("=" * 70)
    print(f"📁 数据集文件: {args.output}")
    print(f"📁 统计信息: {stats_file}")
    print(f"📁 详细日志: dataset_stats.json")


    print("\n" + "=" * 70)
    print("📊 测试集 vs 训练集对比")
    print("=" * 70)
    try:
        with open('train_final_stats.json', 'r') as f:
            train_stats = json.load(f)

        print(f"训练集样本数: {train_stats.get('success', 'N/A')}")
        print(f"测试集样本数: {stats['success']}")
        print(f"测试/训练比例: {stats['success'] / train_stats.get('success', 1) * 100:.1f}%")
    except FileNotFoundError:
        print("(未找到训练集统计信息)")

    print("=" * 70)


if __name__ == "__main__":
    main()