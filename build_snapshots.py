import os, pandas as pd, torch
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset

# ensure raw files exist
ds = EllipticBitcoinDataset(root="data/elliptic")
raw_dir = os.path.join("data/elliptic", "raw")

feats = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_features.csv"), header=None)
edges = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_edgelist.csv"))
classes = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_classes.csv"))

# col 0 = txId, col 1 = timestep, cols 2..166 = features (165 dims)
feats.columns = ["txId", "ts"] + [f"f{i}" for i in range(165)]

# map class strings: '1'=illicit, '2'=licit, 'unknown'
cls_map = {"1": 1, "2": 0, "unknown": -1}
classes["y"] = classes["class"].map(cls_map)

# join features + labels on txId
df = feats.merge(classes[["txId", "y"]], on="txId", how="left")
df["y"] = df["y"].fillna(-1).astype(int)

# global txId -> global index
id2idx = {tx: i for i, tx in enumerate(df["txId"].values)}

# build edges with global indices
edges["src"] = edges["txId1"].map(id2idx)
edges["dst"] = edges["txId2"].map(id2idx)
edges = edges.dropna().astype({"src": int, "dst": int})

# per-snapshot construction
snapshots = []
feat_cols = [f"f{i}" for i in range(165)]
for t in sorted(df["ts"].unique()):
    sub = df[df["ts"] == t]
    global_idx = sub.index.values  # positions in df == global indices
    local_map = {g: l for l, g in enumerate(global_idx)}

    # edges where BOTH endpoints belong to this timestep
    e = edges[edges["src"].isin(local_map) & edges["dst"].isin(local_map)]
    src = e["src"].map(local_map).values
    dst = e["dst"].map(local_map).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    x = torch.tensor(sub[feat_cols].values, dtype=torch.float)
    y = torch.tensor(sub["y"].values, dtype=torch.long)
    labeled_mask = y >= 0  # exclude unknowns

    snap = Data(x=x, edge_index=edge_index, y=y, labeled_mask=labeled_mask, ts=int(t))
    snapshots.append(snap)

torch.save(snapshots, "snapshots.pt")

# quick inspection
print(f"Built {len(snapshots)} snapshots")
for t in [0, 10, 20, 33, 34, 48]:
    s = snapshots[t]
    n_ill = int((s.y == 1).sum()); n_lic = int((s.y == 0).sum())
    print(f"  ts={s.ts:2d}: nodes={s.num_nodes:5d}  edges={s.num_edges:6d}  "
          f"illicit={n_ill:4d}  licit={n_lic:5d}")