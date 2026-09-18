import torch
import esm
import os
import numpy as np
import pickle
from tqdm import tqdm
import biotite.structure as struc
from biotite.structure.io import pdb
import warnings

warnings.filterwarnings("ignore")



TASKS = [





    {
        "name": "Test Set (测试集)",
        "pdb_dir": "/home/lihaotian/new_ec/test_30-50_structures/",
        "output_path": "./30-50test_structure_embeddings_esm_if.pkl"
    }
]

LOCAL_MODEL_PATH = "/home/lihaotian/new_ec/esm_if1_gvp4_t16_142M_UR50.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'UNK': 'X', 'SEC': 'U', 'PYL': 'O'
}


def to_one_letter_manual(res_names):
    seq_list = []
    for r in res_names:
        key = str(r).strip().upper()
        seq_list.append(AA_MAP.get(key, 'X'))
    return "".join(seq_list)


def load_structure(fpath):
    try:
        structure = pdb.PDBFile.read(fpath)
        if structure.get_model_count() == 0: return None, None

        atom_array = pdb.get_structure(structure, model=1)
        atom_array = atom_array[struc.filter_amino_acids(atom_array)]

        try:
            phi, psi, omega = struc.dihedral_backbone(atom_array)
            n_coords = atom_array[atom_array.atom_name == "N"].coord
            ca_coords = atom_array[atom_array.atom_name == "CA"].coord
            c_coords = atom_array[atom_array.atom_name == "C"].coord
            min_len = min(len(n_coords), len(ca_coords), len(c_coords))
            coords = np.stack([n_coords[:min_len], ca_coords[:min_len], c_coords[:min_len]], axis=1)
            res_names = atom_array[atom_array.atom_name == "CA"].res_name[:min_len]
            seq = to_one_letter_manual(res_names)
        except:

            ca_coords = atom_array[atom_array.atom_name == "CA"].coord
            coords = np.stack([ca_coords, ca_coords, ca_coords], axis=1)
            res_names = atom_array[atom_array.atom_name == "CA"].res_name
            seq = to_one_letter_manual(res_names)

        return coords, seq

    except Exception:
        return None, None


def process_directory(model, task_conf):
    pdb_dir = task_conf["pdb_dir"]
    output_path = task_conf["output_path"]
    name = task_conf["name"]

    print(f"\n🔄 开始处理: {name} (无截断，保留完整结构)")

    if not os.path.exists(pdb_dir):
        print(f"   ❌ 错误: 文件夹不存在 {pdb_dir}")
        return

    all_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    print(f"   📊 发现 {len(all_files)} 个 PDB 文件")

    embeddings = {}
    success_count = 0
    fail_count = 0

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for fname in tqdm(all_files, desc=f"提取 {name}"):
            pdb_id = os.path.splitext(fname)[0]
            fpath = os.path.join(pdb_dir, fname)

            coords, seq = load_structure(fpath)

            if coords is None or len(coords) == 0:
                if fail_count < 3: print(f"\n⚠️ 解析失败: {fname}")
                fail_count += 1
                continue

            try:

                coords_t = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(DEVICE)


                padding_mask = torch.zeros((1, coords.shape[0]), dtype=torch.bool).to(DEVICE)



                confidence = torch.ones((1, coords.shape[0]), dtype=torch.float32).to(DEVICE)


                res = model.encoder.forward(
                    coords_t,
                    encoder_padding_mask=padding_mask,
                    confidence=confidence,
                    return_all_hiddens=False
                )

                rep = res['encoder_out'][0]
                struct_emb = rep.mean(dim=0).cpu().numpy()

                embeddings[pdb_id] = struct_emb
                success_count += 1

            except Exception as e:
                if fail_count < 3: print(f"\n❌ 推理错误 {fname}: {e}")
                fail_count += 1
                continue

    print(f"   💾 保存中... -> {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"   ✅ {name} 完成! (成功: {success_count}, 失败: {fail_count})")


def main():
    print(f"🚀 正在初始化环境 (Device: {DEVICE})...")

    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"📂 发现本地模型文件，正在加载...")
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(LOCAL_MODEL_PATH)
    else:
        print(f"☁️ 未找到本地文件，尝试自动下载...")
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    model = model.to(DEVICE)
    model.eval()

    for task in TASKS:
        process_directory(model, task)

    print("\n" + "=" * 50)
    print("🎉 全部完成！")


if __name__ == "__main__":
    main()