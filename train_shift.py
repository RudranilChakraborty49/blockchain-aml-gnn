import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

device = torch.device("cuda")
data = EllipticBitcoinDataset(root="data/elliptic")[0].to(device)

class SAGE(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=2):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, out_dim)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, 0.5, training=self.training)
        return self.c2(x, ei)

def make_random_split(seed=42):
    """Pool all labeled nodes, shuffle, 64/36 split matching temporal sizes."""
    labeled = (data.train_mask | data.test_mask).nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed)
    perm = labeled[torch.randperm(labeled.size(0), generator=g)]
    n_train = int(data.train_mask.sum())  # match temporal split size
    train_idx = perm[:n_train]; test_idx = perm[n_train:]
    train_m = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_m = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_m[train_idx] = True; test_m[test_idx] = True
    return train_m, test_m

def train_eval(train_mask, test_mask, tag):
    torch.manual_seed(42)
    n_ill = int(data.y[train_mask].sum())
    n_lic = int(train_mask.sum()) - n_ill
    weight = torch.tensor([1.0, n_lic / n_ill], device=device)

    model = SAGE(data.num_node_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=weight)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    pred = out.argmax(1); scores = out.softmax(1)[:, 1]
    y = data.y[test_mask].cpu(); p = pred[test_mask].cpu()
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="binary", pos_label=1, zero_division=0)
    prauc = average_precision_score(y, scores[test_mask].cpu())
    return prec, rec, f1, prauc

print(f"{'Split':<12}{'Prec':>8}{'Rec':>8}{'F1':>8}{'PR-AUC':>9}")
print("-"*45)
p,r,f,a = train_eval(data.train_mask, data.test_mask, "temporal")
print(f"{'temporal':<12}{p:>8.3f}{r:>8.3f}{f:>8.3f}{a:>9.3f}")
tr_m, te_m = make_random_split()
p,r,f,a = train_eval(tr_m, te_m, "random")
print(f"{'random':<12}{p:>8.3f}{r:>8.3f}{f:>8.3f}{a:>9.3f}")