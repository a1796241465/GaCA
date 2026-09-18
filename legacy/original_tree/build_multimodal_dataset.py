"""
多模态蛋白质数据集构建脚本
符合论文级标准，包含完整的数据验证和质量控制
"""

import pandas as pd
from Bio.PDB import PDBParser
from Bio import pairwise2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import warnings
import json
from collections import Counter

warnings.filterwarnings('ignore')






def extract_structure_with_validation(pdb_file):
    """
    从PDB文件中提取Cα坐标，并进行严格验证

    为什么这样做：
    1. 只提取标准氨基酸（排除HETATM、水分子等）
    2. 验证坐标合法性（排除NaN、异常值）
    3. 提取PDB实际序列（用于后续对齐）

    Args:
        pdb_file: PDB文件路径

    Returns:
        dict: {
            'coords': np.array (N, 3),
            'residues': List[str],
            'sequence': str,
            'residue_ids': List[int]
        }

    Raises:
        ValueError: 如果无法提取有效的Cα原子
    """

    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        'UNK': 'X'
    }

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    coords = []
    residues_3letter = []
    sequence_1letter = []
    residue_ids = []

    for model in structure:
        for chain in model:
            for residue in chain:



                if residue.id[0] != ' ':
                    continue


                if 'CA' not in residue:
                    continue


                resname = residue.get_resname()


                if resname not in three_to_one:
                    continue


                ca_atom = residue['CA']
                coord = ca_atom.get_coord()


                if np.isnan(coord).any() or np.isinf(coord).any():
                    continue


                if np.abs(coord).max() > 500:
                    continue


                coords.append(coord)
                residues_3letter.append(resname)
                sequence_1letter.append(three_to_one[resname])
                residue_ids.append(residue.id[1])

    if len(coords) == 0:
        raise ValueError("未找到有效的Cα原子")

    return {
        'coords': np.array(coords, dtype=np.float32),
        'residues': residues_3letter,
        'sequence': ''.join(sequence_1letter),
        'residue_ids': residue_ids
    }


def align_sequences(csv_sequence, pdb_sequence):
    """
    对齐CSV序列和PDB序列，找出匹配的区域

    为什么需要对齐：
    - PDB结构可能缺失N端/C端
    - PDB可能包含突变
    - 确保序列和坐标一一对应

    Args:
        csv_sequence: CSV中的完整序列
        pdb_sequence: PDB实际包含的序列

    Returns:
        dict: {
            'csv_start': int,
            'csv_end': int,
            'pdb_start': int,
            'pdb_end': int,
            'identity': float
        }
    """

    alignments = pairwise2.align.globalxx(csv_sequence, pdb_sequence)

    if len(alignments) == 0:
        raise ValueError("序列比对失败")


    best_alignment = alignments[0]
    aligned_csv = best_alignment.seqA
    aligned_pdb = best_alignment.seqB


    csv_start = None
    csv_end = None
    pdb_start = None
    pdb_end = None

    csv_idx = 0
    pdb_idx = 0
    matches = 0
    total = 0

    for i in range(len(aligned_csv)):
        csv_char = aligned_csv[i]
        pdb_char = aligned_pdb[i]


        if csv_char == '-' or pdb_char == '-':
            if csv_char != '-':
                csv_idx += 1
            if pdb_char != '-':
                pdb_idx += 1
            continue


        if csv_start is None:
            csv_start = csv_idx
            pdb_start = pdb_idx


        csv_end = csv_idx + 1
        pdb_end = pdb_idx + 1


        if csv_char == pdb_char:
            matches += 1
        total += 1

        csv_idx += 1
        pdb_idx += 1

    identity = matches / total if total > 0 else 0.0

    return {
        'csv_start': csv_start,
        'csv_end': csv_end,
        'pdb_start': pdb_start,
        'pdb_end': pdb_end,
        'identity': identity
    }


def encode_sequence(sequence):
    """
    将氨基酸序列编码为数字

    为什么这样编码：
    - 标准20种氨基酸映射到0-19
    - 未知氨基酸(X)映射到20
    - 保持与常用数据集的一致性

    Args:
        sequence: 氨基酸序列字符串

    Returns:
        np.array: 编码后的序列 (N,)
    """
    aa_to_idx = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
        'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        'X': 20, 'U': 1, 'O': 8
    }

    encoded = []
    for aa in sequence.upper():
        idx = aa_to_idx.get(aa, 20)
        encoded.append(idx)

    return np.array(encoded, dtype=np.int64)






def validate_sample(sample_data):
    """
    验证单个样本的质量

    验证项目：
    1. 序列长度合理性（100-600）
    2. 坐标范围合理性
    3. EC编号格式正确性
    4. 数据类型正确性

    Args:
        sample_data: 样本字典

    Returns:
        tuple: (is_valid, error_message)
    """

    seq_len = sample_data['sequence_length']
    if seq_len < 50:
        return False, f"序列过短: {seq_len}"
    if seq_len > 1000:
        return False, f"序列过长: {seq_len}"


    coords = sample_data['structure_coords']
    if coords.shape[0] != seq_len:
        return False, f"坐标数量不匹配: {coords.shape[0]} vs {seq_len}"


    coord_min = coords.min()
    coord_max = coords.max()
    if coord_min < -500 or coord_max > 500:
        return False, f"坐标异常: [{coord_min:.1f}, {coord_max:.1f}]"


    ec = sample_data['ec_number']
    parts = ec.split('.')
    if len(parts) != 4:
        return False, f"EC格式错误: {ec}"


    if not isinstance(sample_data['sequence_encoded'], np.ndarray):
        return False, "序列编码类型错误"
    if not isinstance(coords, np.ndarray):
        return False, "坐标类型错误"

    return True, None






def build_multimodal_dataset(
        csv_file='train_cleaned_with_structure.csv',
        structures_dir='structures',
        output_file='multimodal_dataset_clean.pkl',
        min_identity=0.95
):
    """
    构建论文级多模态数据集

    完整流程：
    1. 加载元数据
    2. 逐个处理样本
       a. 提取PDB结构
       b. 对齐序列
       c. 验证质量
    3. 保存数据集和统计信息

    Args:
        csv_file: 清理后的CSV文件
        structures_dir: PDB文件目录
        output_file: 输出pkl文件
        min_identity: 最小序列一致性（推荐0.95）
    """

    print("=" * 70)
    print("🔧 开始构建论文级多模态数据集")
    print("=" * 70)


    df = pd.read_csv(csv_file)
    structures_path = Path(structures_dir)

    print(f"\n【输入数据】")
    print(f"  CSV文件: {csv_file}")
    print(f"  样本数: {len(df)}")
    print(f"  结构目录: {structures_dir}")
    print(f"  最小序列一致性: {min_identity}")


    stats = {
        'total': len(df),
        'success': 0,
        'failed': {
            'pdb_not_found': [],
            'structure_extract_failed': [],
            'alignment_failed': [],
            'low_identity': [],
            'validation_failed': []
        },
        'warnings': {
            'length_adjusted': []
        }
    }

    dataset = []


    print(f"\n【处理样本】")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="进度"):
        uniprot_id = row['Entry']
        pdb_file = structures_path / f"{uniprot_id}.pdb"
        csv_sequence = row['Sequence']

        try:

            if not pdb_file.exists():
                stats['failed']['pdb_not_found'].append(uniprot_id)
                continue


            try:
                structure_data = extract_structure_with_validation(pdb_file)
            except Exception as e:
                stats['failed']['structure_extract_failed'].append({
                    'id': uniprot_id,
                    'error': str(e)
                })
                continue

            pdb_sequence = structure_data['sequence']


            try:
                alignment = align_sequences(csv_sequence, pdb_sequence)
            except Exception as e:
                stats['failed']['alignment_failed'].append({
                    'id': uniprot_id,
                    'error': str(e)
                })
                continue


            if alignment['identity'] < min_identity:
                stats['failed']['low_identity'].append({
                    'id': uniprot_id,
                    'identity': alignment['identity']
                })
                continue


            csv_start = alignment['csv_start']
            csv_end = alignment['csv_end']
            pdb_start = alignment['pdb_start']
            pdb_end = alignment['pdb_end']


            aligned_sequence = csv_sequence[csv_start:csv_end]
            aligned_coords = structure_data['coords'][pdb_start:pdb_end]
            aligned_residues = structure_data['residues'][pdb_start:pdb_end]


            if len(aligned_sequence) != len(csv_sequence):
                stats['warnings']['length_adjusted'].append({
                    'id': uniprot_id,
                    'original': len(csv_sequence),
                    'aligned': len(aligned_sequence)
                })


            sequence_encoded = encode_sequence(aligned_sequence)


            sample = {

                'protein_id': uniprot_id,
                'entry_name': row['Entry Name'],


                'sequence_raw': aligned_sequence,
                'sequence_encoded': sequence_encoded,
                'sequence_length': len(aligned_sequence),


                'structure_coords': aligned_coords,
                'structure_residues': aligned_residues,


                'ec_number': row['EC number'],
                'ec_level1': int(row['EC1']),
                'ec_level2': row['EC2'],
                'ec_level3': row['EC3'],


                'cluster_50': row['clusterRes50'],
                'alignment_identity': alignment['identity']
            }


            is_valid, error = validate_sample(sample)
            if not is_valid:
                stats['failed']['validation_failed'].append({
                    'id': uniprot_id,
                    'error': error
                })
                continue


            dataset.append(sample)
            stats['success'] += 1

        except Exception as e:
            stats['failed']['structure_extract_failed'].append({
                'id': uniprot_id,
                'error': f"Unexpected: {str(e)}"
            })


    print(f"\n【保存数据】")
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"  ✅ 数据集已保存: {output_file}")


    with open('dataset_stats.json', 'w') as f:

        stats_serializable = {
            'total': stats['total'],
            'success': stats['success'],
            'failed': {
                k: [{'id': x['id'], 'error': str(x.get('error', ''))}
                    if isinstance(x, dict) else x
                    for x in v]
                for k, v in stats['failed'].items()
            },
            'warnings': stats['warnings']
        }
        json.dump(stats_serializable, f, indent=2)
    print(f"  ✅ 统计信息已保存: dataset_stats.json")


    print("\n" + "=" * 70)
    print("📊 构建结果统计")
    print("=" * 70)

    print(f"\n【成功率】")
    print(f"  总样本数: {stats['total']}")
    print(f"  成功构建: {stats['success']} ({stats['success'] / stats['total'] * 100:.1f}%)")
    print(f"  失败总数: {stats['total'] - stats['success']}")

    print(f"\n【失败原因分布】")
    for reason, items in stats['failed'].items():
        if len(items) > 0:
            print(f"  {reason}: {len(items)} 个")

    if stats['warnings']['length_adjusted']:
        print(f"\n【警告】")
        print(f"  长度调整: {len(stats['warnings']['length_adjusted'])} 个样本")


    if len(dataset) > 0:
        print(f"\n【数据集质量】")


        seq_lengths = [s['sequence_length'] for s in dataset]
        print(f"  序列长度: {np.min(seq_lengths)}-{np.max(seq_lengths)} aa")
        print(f"  平均长度: {np.mean(seq_lengths):.1f} aa")
        print(f"  中位长度: {np.median(seq_lengths):.0f} aa")


        identities = [s['alignment_identity'] for s in dataset]
        print(f"  平均序列一致性: {np.mean(identities):.3f}")
        print(f"  最低序列一致性: {np.min(identities):.3f}")


        ec_counts = Counter(s['ec_number'] for s in dataset)
        print(f"  唯一EC数: {len(ec_counts)}")
        print(f"  每个EC平均样本数: {len(dataset) / len(ec_counts):.1f}")


        ec1_counts = Counter(s['ec_level1'] for s in dataset)
        print(f"\n【EC Level 1 分布】")
        for ec1 in sorted(ec1_counts.keys()):
            count = ec1_counts[ec1]
            print(f"  EC {ec1}.-.-.-: {count:5d} ({count / len(dataset) * 100:5.1f}%)")


        all_coords = np.concatenate([s['structure_coords'] for s in dataset], axis=0)
        print(f"\n【坐标统计】")
        print(f"  X范围: [{all_coords[:, 0].min():.1f}, {all_coords[:, 0].max():.1f}] Å")
        print(f"  Y范围: [{all_coords[:, 1].min():.1f}, {all_coords[:, 1].max():.1f}] Å")
        print(f"  Z范围: [{all_coords[:, 2].min():.1f}, {all_coords[:, 2].max():.1f}] Å")

    print("\n" + "=" * 70)
    print("✅ 数据集构建完成")
    print("=" * 70)

    return dataset, stats






def test_dataset(pkl_file='multimodal_dataset_clean.pkl'):
    """
    测试数据集的加载和完整性
    """
    print("\n" + "=" * 70)
    print("🧪 测试数据集")
    print("=" * 70)


    print("\n【加载测试】")
    with open(pkl_file, 'rb') as f:
        dataset = pickle.load(f)
    print(f"  ✅ 成功加载 {len(dataset)} 个样本")


    print("\n【样本验证】")
    sample = dataset[0]

    print(f"  蛋白质ID: {sample['protein_id']}")
    print(f"  序列长度: {sample['sequence_length']}")
    print(f"  序列编码shape: {sample['sequence_encoded'].shape}")
    print(f"  结构坐标shape: {sample['structure_coords'].shape}")
    print(f"  EC编号: {sample['ec_number']}")
    print(f"  对齐一致性: {sample['alignment_identity']:.3f}")

    print(f"\n  前10个氨基酸: {sample['sequence_raw'][:10]}")
    print(f"  前3个残基: {sample['structure_residues'][:3]}")
    print(f"  前3个坐标:")
    for i, coord in enumerate(sample['structure_coords'][:3]):
        print(f"    残基{i + 1}: [{coord[0]:7.3f}, {coord[1]:7.3f}, {coord[2]:7.3f}]")


    print("\n【完整性验证】")
    all_valid = True
    for i, s in enumerate(dataset):
        is_valid, error = validate_sample(s)
        if not is_valid:
            print(f"  ❌ 样本{i} ({s['protein_id']}): {error}")
            all_valid = False

    if all_valid:
        print("  ✅ 所有样本验证通过")

    print("\n" + "=" * 70)






if __name__ == "__main__":

    dataset, stats = build_multimodal_dataset(
        csv_file='train_cleaned_with_structure.csv',
        structures_dir='structures',
        output_file='multimodal_dataset_clean.pkl',
        min_identity=0.95
    )


    test_dataset('multimodal_dataset_clean.pkl')

    print("\n💡 下一步:")
    print("  1. 查看 dataset_stats.json 了解详细统计")
    print("  2. 开始构建EGNN数据加载器")
    print("  3. 训练模型")