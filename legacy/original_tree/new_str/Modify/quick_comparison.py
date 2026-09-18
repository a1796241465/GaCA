"""
Quick comparison: GaCA vs ProteinF3S-style vs SES-Adapter-style
Reduced epochs for speed, same architecture as full script.
"""
import os, pickle, json, random, copy, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH, LR, WD, HID, DO = 128, 1e-4, 5e-3, 512, 0.5
EPOCHS, PAT, VAL_R, SEED, LS = 50, 15, 0.1, 42, 0.1
SEQ_D, STR_D = 1280, 512

SEQ_TRAIN = "D:/EC/Sequences_Embeddings_use_ESM-2/Modify/1024_train_seq_embeddings_esm2.pkl"
SEQ_TEST  = "D:/EC/Sequences_Embeddings_use_ESM-2/Modify/1024_test_seq_embeddings_esm2.pkl"
SEQ_T50   = "D:/EC/Sequences_Embeddings_use_ESM-2/Modify/30_50_seq_embeddings_esm2.pkl"
STR_TRAIN = "D:/EC/new_str/Modify/train_structure_embeddings_esm_if.pkl"
STR_TEST  = "D:/EC/new_str/Modify/test_structure_embeddings_esm_if.pkl"
STR_T50   = "D:/EC/new_str/Modify/30-50test_structure_embeddings_esm_if.pkl"
TR_CSV = "D:/EC/train_cleaned_with_structure.csv"
T30_CSV = "D:/EC/test_30_cleaned_with_structure.csv"
T50_CSV = "D:/EC/test_30_50_clean.csv"

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def lp(p):
    with open(p, 'rb') as f: return pickle.load(f)

def pc(batch):
    ss = [b[0] for b in batch]; ts = [b[1] for b in batch]
    return pad_sequence(ss, batch_first=True), pad_sequence(ts, batch_first=True), torch.cat([b[2] for b in batch])

class DS(Dataset):
    def __init__(self, cp, sp, tp, train=True, le=None):
        df = pd.read_csv(cp); df['Entry'] = df['Entry'].astype(str)
        sd, td = lp(sp), lp(tp)
        cm = set(df['Entry']) & set(sd.keys()) & set(td.keys())
        self.id, self.s, self.t, self.lb = [], [], [], []
        for _, r in df.iterrows():
            p = str(r['Entry'])
            if p in cm:
                self.id.append(p); self.s.append(sd[p]); self.t.append(td[p])
                self.lb.append(str(r['EC number']))
        if train: self.le = LabelEncoder(); self.en = self.le.fit_transform(self.lb)
        else:
            self.le = le; kn = set(le.classes_)
            kp = [i for i,L in enumerate(self.lb) if L in kn]
            self.id=[self.id[i] for i in kp]; self.s=[self.s[i] for i in kp]
            self.t=[self.t[i] for i in kp]; self.lb=[self.lb[i] for i in kp]
            self.en = le.transform(self.lb)
    def __len__(self): return len(self.en)
    def __getitem__(self, i):
        s = torch.tensor(self.s[i], dtype=torch.float32)
        t = torch.tensor(self.t[i], dtype=torch.float32)
        if s.dim()>1: s = s.view(-1, SEQ_D)
        if t.dim()>1: t = t.view(-1, STR_D)
        if s.dim()==1: s = s.unsqueeze(0)
        if t.dim()==1: t = t.unsqueeze(0)
        return s, t, torch.LongTensor([self.en[i]])


class AttnPool(nn.Module):
    def __init__(self, d): super().__init__(); self.a = nn.Sequential(nn.Linear(d,128), nn.Tanh(), nn.Linear(128,1))
    def forward(self, x): w = F.softmax(self.a(x), dim=1); return torch.sum(w*x, dim=1)

class GaCA_M(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.sp = AttnPool(SEQ_D); self.tp = AttnPool(STR_D)
        self.spr = nn.Sequential(nn.Linear(SEQ_D,HID), nn.LayerNorm(HID), nn.Dropout(DO))
        self.tpr = nn.Sequential(nn.Linear(STR_D,HID), nn.LayerNorm(HID), nn.Dropout(DO))
        self.gate = nn.Sequential(nn.Linear(HID*2,HID), nn.ReLU(), nn.Linear(HID,HID), nn.Sigmoid())
        self.cls = nn.Sequential(nn.Linear(HID,512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(DO), nn.Linear(512,nc))
    def forward(self, s, t):
        sh = self.spr(self.sp(s)); th = self.tpr(self.tp(t))
        g = self.gate(torch.cat([sh,th], dim=1))
        return self.cls(g*th + (1-g)*sh)

class CAFusion(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.s2t = nn.MultiheadAttention(d, h, batch_first=True, dropout=DO)
        self.t2s = nn.MultiheadAttention(d, h, batch_first=True, dropout=DO)
        self.sg = nn.Sequential(nn.Linear(d*2,d), nn.Sigmoid())
        self.tg = nn.Sequential(nn.Linear(d*2,d), nn.Sigmoid())
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
    def forward(self, s, t):
        se,_ = self.s2t(query=s, key=t, value=t)
        s = self.n1(s + self.sg(torch.cat([s,se],-1))*se)
        te,_ = self.t2s(query=t, key=s, value=s)
        t = self.n2(t + self.tg(torch.cat([t,te],-1))*te)
        return s, t

class ProtF3S_M(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.spr = nn.Sequential(nn.Linear(SEQ_D,HID), nn.LayerNorm(HID), nn.Dropout(DO))
        self.tpr = nn.Sequential(nn.Linear(STR_D,HID), nn.LayerNorm(HID), nn.Dropout(DO))
        self.ca = CAFusion(HID)
        self.sp = AttnPool(HID); self.tp = AttnPool(HID)
        self.fus = nn.Sequential(nn.Linear(HID*2,HID), nn.LayerNorm(HID), nn.GELU(), nn.Dropout(DO))
        self.cls = nn.Sequential(nn.Linear(HID,512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(DO), nn.Linear(512,nc))
    def forward(self, s, t):
        s = self.spr(s); t = self.tpr(t)
        s, t = self.ca(s, t)
        f = self.fus(torch.cat([self.sp(s), self.tp(t)], dim=1))
        return self.cls(f)

class StrAdapter(nn.Module):
    def __init__(self, d, r=4):
        super().__init__()
        self.dn = nn.Linear(d, d//r); self.up = nn.Linear(d//r, d)
        self.sc = nn.Sequential(nn.Linear(d, d//r), nn.GELU())
        self.nm = nn.LayerNorm(d)
    def forward(self, x, c):
        h = self.dn(x) * self.sc(c); h = F.gelu(h); h = self.up(h)
        return self.nm(x + h)

class SESAdp_M(nn.Module):
    def __init__(self, nc, na=3):
        super().__init__()
        self.spr = nn.Sequential(nn.Linear(SEQ_D,HID), nn.LayerNorm(HID), nn.Dropout(DO))
        self.tpr = nn.Sequential(nn.Linear(STR_D,HID), nn.LayerNorm(HID), nn.Dropout(DO))
        self.tp = AttnPool(STR_D)
        self.ads = nn.ModuleList([StrAdapter(HID) for _ in range(na)])
        self.sp = AttnPool(HID)
        self.cls = nn.Sequential(nn.Linear(HID,512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(DO), nn.Linear(512,nc))
    def forward(self, s, t):
        s = self.spr(s); t = self.tpr(t)
        tc = self.tpr[0](self.tp(t))
        for ad in self.ads: s = ad(s, tc)
        return self.cls(self.sp(s))


def compute_mets(yp, yt, pr, le):
    n = len(yt); l1=l2=l3=l4=0
    ts = le.inverse_transform(yt); ps = le.inverse_transform(yp)
    for t,p in zip(ts,ps):
        tp,pp = t.split('.'), p.split('.')
        if len(pp)>=1 and tp[0]==pp[0]: l1+=1
        if len(pp)>=2 and tp[:2]==pp[:2]: l2+=1
        if len(pp)>=3 and tp[:3]==pp[:3]: l3+=1
        if t==p: l4+=1
    mf = f1_score(yt, yp, average='macro')
    uf = f1_score(yt, yp, average='micro')
    mc = matthews_corrcoef(yt, yp)
    nc = len(le.classes_)
    lb = label_binarize(yt, classes=range(nc))
    ap = [average_precision_score(lb[:,i], np.array(pr)[:,i]) for i in range(nc) if np.sum(lb[:,i])>0]
    return {"L1":l1/n,"L2":l2/n,"L3":l3/n,"L4":l4/n,"MacroF1":mf,"MicroF1":uf,"MCC":mc,"AUPR":np.mean(ap) if ap else 0}

def train_one(Model, name, tr_ds, split, t30_ds, t50_ds, le, nc):
    seed_all(SEED)
    m = Model(nc).to(DEVICE)
    np_ = sum(p.numel() for p in m.parameters() if p.requires_grad)
    tr_dl = DataLoader(Subset(tr_ds, split[0]), batch_size=BATCH, shuffle=True, collate_fn=pc)
    vl_dl = DataLoader(Subset(tr_ds, split[1]), batch_size=BATCH, shuffle=False, collate_fn=pc)
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    crit = nn.CrossEntropyLoss(label_smoothing=LS)
    bva, pcnt, bs = 0, 0, None
    for ep in range(EPOCHS):
        m.train()
        for s,t,l in tr_dl:
            s,t,l = s.to(DEVICE), t.to(DEVICE), l.to(DEVICE)
            opt.zero_grad(); crit(m(s,t),l).backward(); opt.step()
        m.eval(); pr, ac = [], []
        with torch.no_grad():
            for s,t,l in vl_dl:
                s,t = s.to(DEVICE), t.to(DEVICE)
                pr.extend(m(s,t).argmax(1).cpu().numpy()); ac.extend(l.numpy())
        va = accuracy_score(ac, pr)
        if va > bva: bva = va; pcnt = 0; bs = copy.deepcopy(m.state_dict())
        else:
            pcnt += 1
            if pcnt >= PAT: break
    m.load_state_dict(bs)
    res = {}
    for ds, nm in [(t30_ds,"<30%"), (t50_ds,"30-50%")]:
        if ds is None: continue
        dl = DataLoader(ds, batch_size=BATCH, shuffle=False, collate_fn=pc)
        m.eval(); yp, yt, pr = [], [], []
        with torch.no_grad():
            for s,t,l in dl:
                s,t = s.to(DEVICE), t.to(DEVICE)
                o = m(s,t); pb = F.softmax(o,dim=1).cpu().numpy()
                yp.extend(o.argmax(1).cpu().numpy()); yt.extend(l.numpy()); pr.extend(pb)
        res[nm] = compute_mets(yp, yt, pr, le)
    return {"name":name,"params":np_,"res":res}


print("="*60)
print("Quick Comparison: GaCA vs ProteinF3S vs SES-Adapter (50 epochs)")
print(f"Device: {DEVICE}")
print("="*60)

print("\nLoading data...")
tr_ds = DS(TR_CSV, SEQ_TRAIN, STR_TRAIN, train=True)
le = tr_ds.le; nc = len(le.classes_)
t30 = DS(T30_CSV, SEQ_TEST, STR_TEST, train=False, le=le)
t50 = DS(T50_CSV, SEQ_T50, STR_T50, train=False, le=le) if os.path.exists(T50_CSV) else None
print(f"Train:{len(tr_ds)} Test<30%:{len(t30)} Test30-50%:{len(t50) if t50 else 'N/A'} Classes:{nc}")

ti, vi = train_test_split(range(len(tr_ds)), test_size=VAL_R, random_state=SEED)
sp = (ti, vi)

all_res = []
for Mdl, nm in [(GaCA_M,"GaCA (Ours)"), (ProtF3S_M,"ProteinF3S-style"), (SESAdp_M,"SES-Adapter-style")]:
    print(f"\n>>> Training {nm} ...")
    r = train_one(Mdl, nm, tr_ds, sp, t30, t50, le, nc)
    all_res.append(r)

    for ds_n in ["<30%", "30-50%"]:
        if ds_n in r["res"]:
            m = r["res"][ds_n]
            print(f"  {ds_n}: L4={m['L4']*100:.2f}% MacroF1={m['MacroF1']:.4f} MCC={m['MCC']:.4f} AUPR={m['AUPR']:.4f}")

print("\n"+"="*70)
print("FINAL COMPARISON TABLE")
print("="*70)
for ds_n in ["<30%", "30-50%"]:
    print(f"\n {ds_n} Sequence Identity:")
    print(f" {'Model':<22} {'L4':<8} {'MacroF1':<10} {'MicroF1':<10} {'MCC':<8} {'AUPR':<8}")
    print(f" {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in all_res:
        m = r["res"].get(ds_n)
        if m is None: continue
        print(f" {r['name']:<22} {m['L4']*100:>6.2f}% {m['MacroF1']:>8.4f}  {m['MicroF1']:>8.4f}  {m['MCC']:>6.4f}  {m['AUPR']:>6.4f}")


with open("D:/EC/new_str/Modify/comparison_results/quick_comparison.json", 'w') as f:
    json.dump([{**r,"res":{k:{kk:float(vv) for kk,vv in v.items()} for k,v in r["res"].items()}} for r in all_res], f, indent=2)

print("\nResults saved. Done!")
