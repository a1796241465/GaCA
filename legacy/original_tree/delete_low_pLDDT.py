"""
清理低置信度数据（通用版本，支持训练集和测试集）
"""
import pandas as pd
from pathlib import Path
import shutil
import argparse


def clean_low_confidence_data(
        details_file='download_details.csv',
        csv_file='train_final_with_structure.csv',
        structures_dir='structures',
        output_csv='train_cleaned_with_structure.csv',
        confidence_threshold=70,
        backup=True
):
    """
    删除低置信度的样本及其对应的PDB文件

    Args:
        details_file: 下载详情CSV
        csv_file: 样本信息CSV
        structures_dir: PDB文件目录
        output_csv: 输出的清理后CSV文件名
        confidence_threshold: 置信度阈值（默认70）
        backup: 是否备份删除的文件
    """

    print("=" * 60)
    print("🧹 开始清理低置信度数据")
    print("=" * 60)


    details = pd.read_csv(details_file)
    df = pd.read_csv(csv_file)
    structures_path = Path(structures_dir)

    print(f"输入CSV: {csv_file}")
    print(f"原始样本数: {len(df)}")
    print(f"置信度阈值: >={confidence_threshold}")


    low_conf_samples = details[details['confidence'] < confidence_threshold]
    low_conf_ids = set(low_conf_samples['uniprot_id'].tolist())

    print(f"\n发现 {len(low_conf_ids)} 个低置信度样本:")
    if len(low_conf_ids) > 0:
        for idx, row in low_conf_samples.iterrows():
            print(f"  {row['uniprot_id']}: confidence={row['confidence']:.1f}")
    else:
        print("  无低置信度样本")


    if backup and len(low_conf_ids) > 0:
        backup_dir = Path('backup_low_confidence_test')
        backup_dir.mkdir(exist_ok=True)

        print(f"\n📦 备份到: {backup_dir}/")
        for uid in low_conf_ids:
            pdb_file = structures_path / f"{uid}.pdb"
            if pdb_file.exists():
                shutil.copy(pdb_file, backup_dir / f"{uid}.pdb")


        low_conf_samples.to_csv(backup_dir / 'low_confidence_samples.csv', index=False)


    print(f"\n🗑️ 删除PDB文件...")
    deleted_count = 0
    for uid in low_conf_ids:
        pdb_file = structures_path / f"{uid}.pdb"
        if pdb_file.exists():
            pdb_file.unlink()
            deleted_count += 1

    print(f"  已删除 {deleted_count} 个PDB文件")


    print(f"\n📝 更新CSV文件...")
    df_cleaned = df[~df['Entry'].isin(low_conf_ids)]


    df_cleaned.to_csv(output_csv, index=False)

    print(f"  原始样本: {len(df)}")
    print(f"  清理后: {len(df_cleaned)}")
    print(f"  保存到: {output_csv}")


    details_cleaned = details[details['confidence'] >= confidence_threshold]
    output_details = output_csv.replace('.csv', '_details.csv')
    details_cleaned.to_csv(output_details, index=False)


    print("\n" + "=" * 60)
    print("📊 清理结果")
    print("=" * 60)
    print(f"删除样本: {len(low_conf_ids)}")
    print(f"保留样本: {len(df_cleaned)}")
    print(f"保留率: {len(df_cleaned) / len(df) * 100:.1f}%")


    if len(details_cleaned) > 0:
        remaining_conf = details_cleaned['confidence']
        print(f"\n清理后质量分布:")
        print(f"  平均置信度: {remaining_conf.mean():.1f}")
        print(f"  最低置信度: {remaining_conf.min():.1f}")
        print(f"  最高置信度: {remaining_conf.max():.1f}")
        print(
            f"  >90分: {(remaining_conf > 90).sum()} 个 ({(remaining_conf > 90).sum() / len(remaining_conf) * 100:.1f}%)")

    return df_cleaned, low_conf_ids



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='清理低置信度数据')
    parser.add_argument('--details', type=str, default='download_details.csv',
                        help='下载详情CSV文件')
    parser.add_argument('--csv', type=str, default='train_final_with_structure.csv',
                        help='输入CSV文件')
    parser.add_argument('--structures', type=str, default='structures',
                        help='PDB文件目录')
    parser.add_argument('--output', type=str, default='train_cleaned_with_structure.csv',
                        help='输出CSV文件')
    parser.add_argument('--threshold', type=int, default=70,
                        help='置信度阈值')

    args = parser.parse_args()

    df_cleaned, removed_ids = clean_low_confidence_data(
        details_file=args.details,
        csv_file=args.csv,
        structures_dir=args.structures,
        output_csv=args.output,
        confidence_threshold=args.threshold,
        backup=True
    )

    print("\n✅ 清理完成！")
    print(f"   清理后的CSV: {args.output}")
    print("   下一步: 运行构建数据集脚本")