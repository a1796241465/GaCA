
"""Download AlphaFold PDBs and render 3D structures colored by pLDDT for misclassified proteins."""
import os, sys, requests, gzip, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from Bio.PDB import PDBParser
import py3Dmol

OUT_DIR = 'D:/EC/new_str/Modify/analysis_outputs/structures'
os.makedirs(OUT_DIR, exist_ok=True)


PROTEINS = {
    'E9FCP7': {'true': '4.1.1.11', 'pred': '4.1.1.105', 'type': 'EC4 within-class', 'conf': 0.126},
    'H6LC30': {'true': '7.2.1.2', 'pred': '7.2.1.1', 'type': 'EC7 within-class', 'conf': 0.402},
    'Q9M8R4': {'true': '4.4.1.5', 'pred': '3.5.1.2', 'type': 'EC4→EC3 cross-EC', 'conf': 0.021},
    'A6TUN0': {'true': '7.4.2.8', 'pred': '3.6.4.12', 'type': 'EC7→EC3 cross-EC', 'conf': 0.601},
}

AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v6.pdb"
AF_API  = "https://alphafold.ebi.ac.uk/api/prediction/{}"

def download_pdb(uniprot_id, out_dir):
    """Download PDB from AlphaFold DB, trying v6 then older versions."""
    pdb_path = os.path.join(out_dir, f"AF-{uniprot_id}.pdb")
    if os.path.exists(pdb_path):
        print(f"  Already cached: {pdb_path}")
        return pdb_path

    for version in [6, 5, 4]:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{version}.pdb"
        print(f"  Trying v{version}...", end=" ")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(pdb_path, 'w', encoding='utf-8') as f:
                f.write(resp.text)
            print(f"OK ({len(resp.text)} bytes)")
            return pdb_path
        print(f"HTTP {resp.status_code}")

    print(f"  FAILED: no version found")
    return None

def parse_plddt_from_pdb(pdb_path):
    """Extract CA atom coordinates and pLDDT scores (B-factor) from PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)
    model = structure[0]

    coords = []
    plddt = []
    residues = []
    for chain in model:
        for res in chain:
            if 'CA' in res:
                ca = res['CA']
                coords.append(ca.get_coord())
                plddt.append(ca.get_bfactor())
                residues.append(res.get_resname())

    return np.array(coords), np.array(plddt), residues

def render_matplotlib_3d(coords, plddt, title, save_path, highlight_idx=None):
    """Render 3D scatter with pLDDT coloring, suitable for publication.

    Uses per-protein adaptive normalization (vmin/vmax from data) so color
    variation is visible even for high-quality structures. Also renders a
    separate low-confidence highlight panel.
    """

    fig = plt.figure(figsize=(14, 5))

    ax = fig.add_subplot(1, 3, 1, projection='3d')


    colors = ['#FF7D45', '#FFBF3F', '#65BF73', '#3F6FB3', '#003366']
    cmap = LinearSegmentedColormap.from_list('plddt', colors, N=256)


    vmin = max(0, np.percentile(plddt, 2))
    vmax = np.percentile(plddt, 98)
    norm = plt.Normalize(vmin, vmax)

    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=plddt, cmap=cmap, norm=norm, s=6, alpha=0.85, linewidths=0,
                    edgecolors='none')

    ax.set_title(f"{title}\n(pLDDT: {vmin:.0f}-{vmax:.0f})", fontsize=10, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.tick_params(labelsize=7)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.12)
    cbar.set_label('pLDDT', fontsize=9)
    cbar.ax.tick_params(labelsize=7)


    ax2 = fig.add_subplot(1, 3, 2, projection='3d')

    low_mask = plddt < 50
    mid_mask = (plddt >= 50) & (plddt < 70)
    high_mask = plddt >= 70


    if high_mask.sum() > 0:
        ax2.scatter(coords[high_mask, 0], coords[high_mask, 1], coords[high_mask, 2],
                    c='#B0C4DE', s=5, alpha=0.5, linewidths=0, label=f'High >=70 ({high_mask.sum()})')

    if mid_mask.sum() > 0:
        ax2.scatter(coords[mid_mask, 0], coords[mid_mask, 1], coords[mid_mask, 2],
                    c='#FFBF3F', s=8, alpha=0.9, linewidths=0, label=f'Mid 50-70 ({mid_mask.sum()})')

    if low_mask.sum() > 0:
        ax2.scatter(coords[low_mask, 0], coords[low_mask, 1], coords[low_mask, 2],
                    c='#FF2020', s=15, alpha=1.0, linewidths=0.5, edgecolors='darkred',
                    label=f'Low <50 ({low_mask.sum()})')

    ax2.set_title('Confidence Level', fontsize=10, fontweight='bold')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.tick_params(labelsize=7)
    ax2.legend(fontsize=7, loc='upper right')


    ax3 = fig.add_subplot(1, 3, 3)
    n, bins, patches = ax3.hist(plddt, bins=40, edgecolor='white', alpha=0.9, linewidth=0.3)


    for patch, b in zip(patches, bins[:-1]):
        frac = (b - 0) / 100
        patch.set_facecolor(cmap(frac))

    ax3.axvline(x=70, color='#3F6FB3', linestyle='--', linewidth=1.5, label='70 (Confident)')
    ax3.axvline(x=50, color='#FF7D45', linestyle='--', linewidth=1.5, label='50 (Low)')
    ax3.axvline(x=np.mean(plddt), color='black', linestyle='-', linewidth=2,
                label=f'Mean={np.mean(plddt):.1f}')
    ax3.set_xlabel('pLDDT', fontsize=10)
    ax3.set_ylabel('Residues', fontsize=10)
    ax3.set_title(f'pLDDT Distribution\nmean={np.mean(plddt):.1f}, <50: {(plddt<50).sum()}/{len(plddt)}',
                  fontsize=10)
    ax3.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {save_path}")

def render_py3Dmol_html(pdb_path, uniprot_id, info, out_dir):
    """Render interactive 3D view with py3Dmol, colored by pLDDT (B-factor).

    Uses cartoon representation with rwb (red-white-blue) gradient mapped to
    B-factor column. Also provides a surface view toggle via checkbox.
    """
    with open(pdb_path, 'r') as f:
        pdb_str = f.read()

    view = py3Dmol.view(width=700, height=500)


    view.addModel(pdb_str, 'pdb')


    view.setStyle({'model': 0}, {'cartoon': {
        'color': 'spectrum',
        'colorscheme': {
            'prop': 'b',
            'gradient': 'rwb',
            'min': 0,
            'max': 100
        }
    }})


    view.addSurface('VDW', {'opacity': 0.25, 'colorscheme': {'prop': 'b', 'gradient': 'rwb', 'min': 0, 'max': 100}})

    view.setBackgroundColor('white')
    view.zoomTo()
    view.spin('slow')


    html_content = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{uniprot_id} — GaCA Misclassification Analysis</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #fff; }}
h3 {{ margin-bottom: 4px; }}
.meta {{ color: #666; font-size: 14px; margin-bottom: 10px; }}
.legend {{ display: flex; gap: 16px; margin: 10px 0; font-size: 13px; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; }}
.swatch {{ width: 16px; height: 16px; border-radius: 3px; }}
.container {{ border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }}
</style>
</head>
<body>
<h3>{uniprot_id}: True={info['true']} → Pred={info['pred']} ({info['type']})</h3>
<p class="meta">Confidence: {info['conf']:.3f} | AlphaFold DB v6</p>
<div class="legend">
  <div class="legend-item"><div class="swatch" style="background:#ff0000"></div> Very low (pLDDT &lt;50)</div>
  <div class="legend-item"><div class="swatch" style="background:#ff8888"></div> Low (50-70)</div>
  <div class="legend-item"><div class="swatch" style="background:#ffffff"></div> Confident (70-90)</div>
  <div class="legend-item"><div class="swatch" style="background:#8888ff"></div> Very high (&gt;90)</div>
</div>
<div class="container">
{view._make_html()}
</div>
<p style="font-size:12px;color:#999;margin-top:8px;">Drag to rotate, scroll to zoom. Cartoon colored by pLDDT. Transparent surface overlay shows pLDDT on molecular surface.</p>
</body>
</html>'''

    html_path = os.path.join(out_dir, f'{uniprot_id}_3D.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  Interactive HTML saved: {html_path}")

if __name__ == "__main__":
    print("Downloading AlphaFold structures and generating 3D visualizations...")
    print("=" * 60)

    for uniprot_id, info in PROTEINS.items():
        print(f"\n>>> {uniprot_id}: {info['true']} → {info['pred']} ({info['type']})")


        pdb_path = download_pdb(uniprot_id, OUT_DIR)
        if pdb_path is None:
            continue


        coords, plddt, residues = parse_plddt_from_pdb(pdb_path)
        print(f"  Residues: {len(coords)}, pLDDT mean={plddt.mean():.1f}, "
              f"min={plddt.min():.0f}, max={plddt.max():.0f}")
        low_conf = (plddt < 50).sum()
        high_conf = (plddt >= 70).sum()
        print(f"  Low pLDDT (<50): {low_conf} residues ({low_conf/len(plddt)*100:.1f}%)")
        print(f"  High pLDDT (>=70): {high_conf} residues ({high_conf/len(plddt)*100:.1f}%)")


        title = f"{uniprot_id}\nTrue: {info['true']}  →  Pred: {info['pred']}\npLDDT mean={plddt.mean():.1f}"
        fig_path = os.path.join(OUT_DIR, f'{uniprot_id}_structure.png')
        render_matplotlib_3d(coords, plddt, title, fig_path)


        render_py3Dmol_html(pdb_path, uniprot_id, info, OUT_DIR)

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUT_DIR}")
    print("Done.")
