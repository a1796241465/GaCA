import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import time
import json


def get_alphafold_info(uniprot_id):
    """
    通过AlphaFold API获取蛋白质信息和PDB下载链接
    """
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                entry = data[0]

                return {
                    'pdb_url': entry.get('pdbUrl'),
                    'cif_url': entry.get('cifUrl'),
                    'version': entry.get('latestVersion'),
                    'confidence': entry.get('globalMetricValue'),
                    'status': 'success'
                }
            else:
                return {'status': 'empty_response'}

        elif response.status_code == 404:
            return {'status': 'not_in_alphafold'}
        else:
            return {'status': f'api_error_{response.status_code}'}

    except Exception as e:
        return {'status': f'exception: {str(e)}'}


def download_pdb_file(url, save_path, retry=3):
    """
    从指定URL下载PDB文件
    """
    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            elif response.status_code == 404:
                return False
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2)
                continue
            return False
    return False


def batch_download_alphafold_structures(df, save_dir='structures'):
    """
    批量下载AlphaFold结构（最终版）
    """
    Path(save_dir).mkdir(exist_ok=True)

    results = {
        'success': [],
        'exists': [],
        'not_in_alphafold': [],
        'download_failed': [],
        'api_error': [],
        'low_confidence': []
    }


    details = []

    print("\n" + "=" * 60)
    print("📥 开始批量下载AlphaFold结构（最终版 v6）")
    print("=" * 60)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="下载进度"):
        uniprot_id = row['Entry']
        save_path = Path(save_dir) / f"{uniprot_id}.pdb"


        if save_path.exists():
            results['exists'].append(uniprot_id)
            details.append({
                'uniprot_id': uniprot_id,
                'status': 'exists',
                'file_path': str(save_path)
            })
            continue


        info = get_alphafold_info(uniprot_id)

        if info['status'] == 'success' and info.get('pdb_url'):
            pdb_url = info['pdb_url']
            confidence = info.get('confidence', 0)


            if download_pdb_file(pdb_url, save_path):
                results['success'].append(uniprot_id)


                if confidence < 70:
                    results['low_confidence'].append(uniprot_id)

                details.append({
                    'uniprot_id': uniprot_id,
                    'status': 'downloaded',
                    'confidence': confidence,
                    'version': info.get('version'),
                    'file_path': str(save_path)
                })
            else:
                results['download_failed'].append(uniprot_id)
                details.append({
                    'uniprot_id': uniprot_id,
                    'status': 'download_failed',
                    'url': pdb_url
                })

        elif info['status'] == 'not_in_alphafold':
            results['not_in_alphafold'].append(uniprot_id)
            details.append({
                'uniprot_id': uniprot_id,
                'status': 'not_in_alphafold'
            })

        else:
            results['api_error'].append((uniprot_id, info['status']))
            details.append({
                'uniprot_id': uniprot_id,
                'status': info['status']
            })


        if (idx + 1) % 100 == 0:
            time.sleep(2)

            pd.DataFrame(details).to_csv('download_progress.csv', index=False)


    print("\n" + "=" * 60)
    print("📊 下载统计")
    print("=" * 60)
    print(f"✅ 新下载成功: {len(results['success'])} 个")
    print(f"📁 已存在跳过: {len(results['exists'])} 个")
    print(f"❌ AlphaFold中无记录: {len(results['not_in_alphafold'])} 个")
    print(f"⚠️ 下载失败: {len(results['download_failed'])} 个")
    print(f"🔥 API错误: {len(results['api_error'])} 个")

    if results['low_confidence']:
        print(f"⚠️ 低置信度(<70): {len(results['low_confidence'])} 个")

    total_available = len(results['success']) + len(results['exists'])
    success_rate = total_available / len(df) * 100
    print(f"\n📈 总可用结构: {total_available}/{len(df)} ({success_rate:.1f}%)")


    details_df = pd.DataFrame(details)
    details_df.to_csv('download_details.csv', index=False)
    print(f"\n💾 详细结果已保存到: download_details.csv")


    if results['not_in_alphafold'] or results['download_failed'] or results['api_error']:
        failed_data = []

        for uid in results['not_in_alphafold']:
            failed_data.append({'UniProt_ID': uid, 'Reason': 'not_in_alphafold'})

        for uid in results['download_failed']:
            failed_data.append({'UniProt_ID': uid, 'Reason': 'download_failed'})

        for uid, reason in results['api_error']:
            failed_data.append({'UniProt_ID': uid, 'Reason': reason})

        failed_df = pd.DataFrame(failed_data)
        failed_df.to_csv('download_failed.csv', index=False)
        print(f"💾 失败列表已保存到: download_failed.csv")

    return results, details_df



if __name__ == "__main__":
    print("🔧 AlphaFold结构批量下载工具（v6最终版）")
    print("=" * 60)


    df = pd.read_csv('train_filtered_final.csv')
    print(f"📊 总共需要下载: {len(df)} 个蛋白质结构\n")


    print("🧪 测试前5个样本...")
    test_df = df.head(5)

    for idx, row in test_df.iterrows():
        uniprot_id = row['Entry']
        info = get_alphafold_info(uniprot_id)

        if info['status'] == 'success':
            print(f"✅ {uniprot_id}:")
            print(f"   URL: {info['pdb_url']}")
            print(f"   Version: v{info['version']}")
            print(f"   Confidence: {info['confidence']:.1f}")
        else:
            print(f"❌ {uniprot_id}: {info['status']}")


    print("\n" + "=" * 60)
    choice = input("是否继续下载全部结构？(y/n): ")

    if choice.lower() == 'y':
        results, details = batch_download_alphafold_structures(df)


        available_ids = set(results['success'] + results['exists'])
        df_with_structure = df[df['Entry'].isin(available_ids)]


        output_file = 'train_final_with_structure.csv'
        df_with_structure.to_csv(output_file, index=False)

        print("\n" + "=" * 60)
        print("✅ 所有处理完成！")
        print("=" * 60)
        print(f"📁 最终数据集: {output_file}")
        print(f"   样本数: {len(df_with_structure)}")
        print(f"📁 结构文件目录: structures/")
        print(f"   文件数: {len(list(Path('structures').glob('*.pdb')))}")
        print(f"📁 详细日志: download_details.csv")

        if results['not_in_alphafold']:
            print(f"\n⚠️ 提示: {len(results['not_in_alphafold'])} 个蛋白质在AlphaFold中无记录")
            print(f"   可以考虑使用ESMFold为这些蛋白质预测结构")
    else:
        print("❌ 已取消下载")