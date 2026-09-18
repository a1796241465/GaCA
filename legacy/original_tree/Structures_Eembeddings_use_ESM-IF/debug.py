import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import numpy as np
import os


TEST_PDB = "/home/lihaotian/new_ec/train_structures/A0A011QK89.pdb"


AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'UNK': 'X', 'SEC': 'U', 'PYL': 'O'
}


def to_one_letter_manual(res_names):

    return "".join([AA_MAP.get(str(r).strip(), 'X') for r in res_names])


print(f"正在测试读取: {TEST_PDB}")
try:
    structure = pdb.PDBFile.read(TEST_PDB)
    atom_array = pdb.get_structure(structure, model=1)


    atom_array = atom_array[struc.filter_amino_acids(atom_array)]


    phi, psi, omega = struc.dihedral_backbone(atom_array)

    n_coords = atom_array[atom_array.atom_name == "N"].coord
    ca_coords = atom_array[atom_array.atom_name == "CA"].coord
    c_coords = atom_array[atom_array.atom_name == "C"].coord

    print(f"坐标提取成功: N={len(n_coords)}, CA={len(ca_coords)}, C={len(c_coords)}")

    min_len = min(len(n_coords), len(ca_coords), len(c_coords))
    coords = np.stack([n_coords[:min_len], ca_coords[:min_len], c_coords[:min_len]], axis=1)


    res_names = atom_array[atom_array.atom_name == "CA"].res_name[:min_len]
    print(f"残基名称样本: {res_names[:5]}")

    seq = to_one_letter_manual(res_names)
    print(f"转换序列成功: {seq[:10]}...")

    print("\n✅ 测试通过！看来之前的脚本里可能有些微小的逻辑问题。")

except Exception as e:
    print("\n❌ 捕捉到错误！请把下面这段发给我：")
    print("=" * 30)
    import traceback

    traceback.print_exc()
    print("=" * 30)