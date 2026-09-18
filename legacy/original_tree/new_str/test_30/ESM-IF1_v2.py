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
        "name": "Train Set (训练集)",
        "pdb_dir": "/home/lihaotian/new_ec/train_structures/",
        "output_path": "./train_structure_embeddings_esm_if.pkl"
    },
    {
        "name": "Test Set (测试集)",
        "pdb_dir": "/home/lihaotian/new_ec/test_structures/",
        "output_path": "./30-50test_structure_embeddings_esm_if.pkl"
    }
]

LOCAL_MODEL_PATH = "/home/lihaotian/new_ec/esm_if1_gvp4_t16_142M_UR50.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_structure_backbone(fpath):
    """
    Load backbone coordinates (N, CA, C).
    Returns:
        coords: (L, 3, 3) or None
        is_valid: bool (False if fallback / invalid)
    """
    try:
        structure = pdb.PDBFile.read(fpath)
        if structure.get_model_count() == 0:
            return None, False

        atom_array = pdb.get_structure(structure, model=1)
        atom_array = atom_array[struc.filter_amino_acids(atom_array)]

        n_coords = atom_array[atom_array.atom_name == "N"].coord
        ca_coords = atom_array[atom_array.atom_name == "CA"].coord
        c_coords = atom_array[atom_array.atom_name == "C"].coord

        min_len = min(len(n_coords), len(ca_coords), len(c_coords))
        if min_len == 0:
            return None, False

        coords = np.stack(
            [n_coords[:min_len], ca_coords[:min_len], c_coords[:min_len]],
            axis=1
        )

        return coords, True

    except Exception:
        return None, False


def extract_structure_embedding(model, coords_np):
    """
    ESM-IF structure encoder (sequence-independent).
    Input:
        coords_np: (L, 3, 3)
    Output:
        protein_emb: (512,)
    """
    coords_t = torch.tensor(coords_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    padding_mask = torch.zeros((1, coords_np.shape[0]), dtype=torch.bool).to(DEVICE)
    confidence = torch.ones((1, coords_np.shape[0]), dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        res = model.encoder.forward(
            coords_t,
            encoder_padding_mask=padding_mask,
            confidence=confidence,
            return_all_hiddens=False
        )

        residue_repr = res["encoder_out"][0]
        protein_repr = residue_repr.mean(dim=0)
        protein_repr = protein_repr / (protein_repr.norm() + 1e-6)

    return protein_repr.cpu().numpy()


def process_directory(model, task_conf):
    pdb_dir = task_conf["pdb_dir"]
    output_path = task_conf["output_path"]
    name = task_conf["name"]

    print(f"\n🔄 开始处理: {name}")
    print("   ➤ 使用 ESM-IF 结构编码器（sequence-independent）")
    print("   ➤ Backbone 原子: N / CA / C")
    print("   ➤ Protein-level mean pooling + L2 norm")

    if not os.path.exists(pdb_dir):
        print(f"❌ 文件夹不存在: {pdb_dir}")
        return

    all_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith(".pdb")])
    print(f"📊 发现 {len(all_files)} 个 PDB 文件")

    embeddings = {}
    bad_structures = []
    success, fail = 0, 0

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(all_files, desc=f"提取 {name}"):
        pdb_id = os.path.splitext(fname)[0]
        fpath = os.path.join(pdb_dir, fname)

        coords, is_valid = load_structure_backbone(fpath)
        if not is_valid:
            bad_structures.append(pdb_id)
            fail += 1
            continue

        try:
            emb = extract_structure_embedding(model, coords)
            embeddings[pdb_id] = emb
            success += 1
        except Exception as e:
            bad_structures.append(pdb_id)
            fail += 1
            if fail <= 3:
                print(f"\n❌ 推理失败 {fname}: {e}")

    print(f"\n📈 成功: {success} | 失败/剔除: {fail}")

    if len(bad_structures) > 0:
        print(f"⚠️ 剔除 {len(bad_structures)} 个无效结构（缺失 backbone）")

    print(f"💾 保存 embedding -> {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"✅ {name} 完成")


def main():
    print(f"🚀 初始化 ESM-IF (Device: {DEVICE})")

    if os.path.exists(LOCAL_MODEL_PATH):
        print("📂 使用本地 ESM-IF 模型")
        model, _ = esm.pretrained.load_model_and_alphabet_local(LOCAL_MODEL_PATH)
    else:
        print("☁️ 下载官方 ESM-IF 模型")
        model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    model = model.to(DEVICE)
    model.eval()

    for task in TASKS:
        process_directory(model, task)

    print("\n" + "=" * 60)
    print("🎉 所有结构 embedding 提取完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
