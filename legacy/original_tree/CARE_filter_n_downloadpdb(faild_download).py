import pandas as pd
import numpy as np
from collections import Counter
import requests
from pathlib import Path
from tqdm import tqdm
import time





def load_and_analyze_data(file_path):
    """加载数据并分析基本统计信息"""
    df = pd.read_csv(file_path)

    print("=" * 60)
    print("📊 数据集基本信息")
    print("=" * 60)
    print(f"总样本数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")
    print(f"\n缺失值统计:")
    print(df.isnull().sum())


    print(f"\n🔬 EC编号统计:")
    print(f"  - EC Level 4 (X.X.X.X): {df['EC number'].nunique()} 个")
    print(f"  - EC Level 3 (X.X.X.-): {df['EC3'].nunique()} 个")
    print(f"  - EC Level 2 (X.X.-.-): {df['EC2'].nunique()} 个")
    print(f"  - EC Level 1 (X.-.-.-): {df['EC1'].nunique()} 个")


    print(f"\n📏 序列长度统计:")
    print(f"  - 最短: {df['Length'].min()} aa")
    print(f"  - 最长: {df['Length'].max()} aa")
    print(f"  - 平均: {df['Length'].mean():.1f} aa")
    print(f"  - 中位数: {df['Length'].median():.1f} aa")

    return df





def filter_by_length(df, max_length=600):
    """按序列长度过滤"""
    filtered_df = df[df['Length'] <= max_length].copy()
    print(f"\n✂️ 长度过滤 (≤{max_length} aa):")
    print(f"  过滤前: {len(df)} 条")
    print(f"  过滤后: {len(filtered_df)} 条")
    print(f"  移除比例: {(1 - len(filtered_df) / len(df)) * 100:.1f}%")
    return filtered_df


def stratified_sampling(df, max_per_ec=30, ec_level='EC number', random_state=42):
    """
    分层采样策略

    参数:
        df: 数据集
        max_per_ec: 每个EC最多保留的样本数
        ec_level: 使用的EC层级 ('EC number', 'EC3', 'EC2', 'EC1')
        random_state: 随机种子
    """
    print(f"\n🎲 分层采样 (每个{ec_level}最多保留{max_per_ec}条):")


    ec_counts = df[ec_level].value_counts()
    print(f"  - 总共 {len(ec_counts)} 个不同的{ec_level}")
    print(f"  - 样本数最多的EC: {ec_counts.iloc[0]} 条")
    print(f"  - 样本数最少的EC: {ec_counts.iloc[-1]} 条")

    sampled_dfs = []

    for ec in df[ec_level].unique():
        ec_df = df[df[ec_level] == ec]

        if len(ec_df) > max_per_ec:

            ec_sampled = ec_df.sample(n=max_per_ec, random_state=random_state)
        else:

            ec_sampled = ec_df

        sampled_dfs.append(ec_sampled)

    result_df = pd.concat(sampled_dfs, ignore_index=True)

    print(f"  采样前: {len(df)} 条")
    print(f"  采样后: {len(result_df)} 条")
    print(f"  保留比例: {len(result_df) / len(df) * 100:.1f}%")


    new_ec_counts = result_df[ec_level].value_counts()
    print(f"\n  📊 采样后的EC分布:")
    print(f"    - 最多样本数: {new_ec_counts.max()} 条")
    print(f"    - 最少样本数: {new_ec_counts.min()} 条")
    print(f"    - 平均样本数: {new_ec_counts.mean():.1f} 条")

    return result_df


def remove_duplicates(df):
    """移除重复的UniProt ID"""
    print(f"\n🔄 去重:")
    print(f"  去重前: {len(df)} 条")
    df_unique = df.drop_duplicates(subset='Entry', keep='first')
    print(f"  去重后: {len(df_unique)} 条")
    return df_unique





def download_alphafold_structure(uniprot_id, save_dir='structures', retry=3):
    """
    下载AlphaFold预测结构

    参数:
        uniprot_id: UniProt ID
        save_dir: 保存目录
        retry: 重试次数
    """
    Path(save_dir).mkdir(exist_ok=True)
    save_path = Path(save_dir) / f"{uniprot_id}.pdb"


    if save_path.exists():
        return 'exists'

    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return 'success'
            elif response.status_code == 404:
                return 'not_found'
            else:
                if attempt < retry - 1:
                    time.sleep(2)
                    continue
                return 'failed'

        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2)
                continue
            return f'error: {str(e)}'

    return 'failed'


def batch_download_structures(df, save_dir='structures', max_concurrent=5):
    """
    批量下载结构文件

    参数:
        df: 包含Entry列的DataFrame
        save_dir: 保存目录
        max_concurrent: 每批次下载数量
    """
    print("\n" + "=" * 60)
    print("📥 开始批量下载AlphaFold结构")
    print("=" * 60)

    results = {
        'success': [],
        'exists': [],
        'not_found': [],
        'failed': [],
        'error': []
    }

    total = len(df)

    for idx, row in tqdm(df.iterrows(), total=total, desc="下载进度"):
        uniprot_id = row['Entry']

        status = download_alphafold_structure(uniprot_id, save_dir)

        if status == 'success':
            results['success'].append(uniprot_id)
        elif status == 'exists':
            results['exists'].append(uniprot_id)
        elif status == 'not_found':
            results['not_found'].append(uniprot_id)
        elif status.startswith('error'):
            results['error'].append((uniprot_id, status))
        else:
            results['failed'].append(uniprot_id)


        if (idx + 1) % 100 == 0:
            time.sleep(3)


    print("\n" + "=" * 60)
    print("📊 下载统计")
    print("=" * 60)
    print(f"✅ 新下载成功: {len(results['success'])} 个")
    print(f"📁 已存在跳过: {len(results['exists'])} 个")
    print(f"❌ AlphaFold中无记录: {len(results['not_found'])} 个")
    print(f"⚠️ 下载失败: {len(results['failed'])} 个")
    print(f"🔥 错误: {len(results['error'])} 个")

    total_available = len(results['success']) + len(results['exists'])
    print(f"\n📈 总可用结构: {total_available}/{total} ({total_available / total * 100:.1f}%)")


    if results['not_found'] or results['failed'] or results['error']:
        failed_df = pd.DataFrame({
            'UniProt_ID': (results['not_found'] + results['failed'] +
                           [x[0] for x in results['error']]),
            'Status': (['not_found'] * len(results['not_found']) +
                       ['failed'] * len(results['failed']) +
                       [x[1] for x in results['error']])
        })
        failed_df.to_csv('download_failed.csv', index=False)
        print(f"\n💾 失败列表已保存到: download_failed.csv")

    return results





def main():
    """完整处理流程"""


    print("\n🔧 Step 1: 加载数据")
    df = load_and_analyze_data('/home/lihaotian/new_ec/CARE_datasets/splits/task1/protein_train50.csv')


    print("\n🔧 Step 2: 序列长度过滤")
    df_filtered = filter_by_length(df, max_length=600)


    print("\n🔧 Step 3: 分层采样")

    df_sampled = stratified_sampling(
        df_filtered,
        max_per_ec=25,
        ec_level='EC number',
        random_state=42
    )


    print("\n🔧 Step 4: 去除重复")
    df_final = remove_duplicates(df_sampled)


    output_file = 'train_filtered_final.csv'
    df_final.to_csv(output_file, index=False)
    print(f"\n💾 筛选后的数据已保存到: {output_file}")


    print("\n" + "=" * 60)
    print("📊 最终数据集统计")
    print("=" * 60)
    print(f"总样本数: {len(df_final)}")
    print(f"EC Level 4 数量: {df_final['EC number'].nunique()}")
    print(f"序列长度范围: {df_final['Length'].min()} - {df_final['Length'].max()} aa")
    print(f"平均序列长度: {df_final['Length'].mean():.1f} aa")


    print("\n🔧 Step 5: 下载AlphaFold结构")
    download_confirm = input("是否开始下载结构文件？(y/n): ")

    if download_confirm.lower() == 'y':
        results = batch_download_structures(df_final, save_dir='structures')


        available_ids = set(results['success'] + results['exists'])
        df_with_structure = df_final[df_final['Entry'].isin(available_ids)]


        df_with_structure.to_csv('train_final_with_structure.csv', index=False)
        print(f"\n💾 有结构的数据已保存到: train_final_with_structure.csv")
        print(f"   最终可用样本: {len(df_with_structure)} 个")

    print("\n✅ 所有处理完成！")


if __name__ == "__main__":
    main()