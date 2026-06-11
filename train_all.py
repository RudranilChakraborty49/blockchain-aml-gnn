import torch
import torch.nn.functional as F
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = EllipticBitcoinDataset(root="data/elliptic")[0].to(device)

n_illicit = int(data.y[data.train_mask].sum())
n_licit = int(data.train_mask.sum()) - n_illicit
weight = torch.tensor([1.0, n_licit / n_illicit], device=device)

def evaluate(pred, scores, mask):
    mask = mask.cpu()
    y = data.y.cpu()[mask]; p = pred.cpu()[mask]
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, p, average="binary", pos_label=1, zero_division=0)
    pr_auc = average_precision_score(y, scores.cpu()[mask])
    return prec, rec, f1, pr_auc

class GNN(torch.nn.Module):
    def __init__(self, conv, in_dim, hidden=128, out_dim=2, **kw):
        super().__init__()
        self.c1 = conv(in_dim, hidden, **kw)
        h2_in = hidden * kw.get("heads", 1) if conv is GATConv else hidden
        self.c2 = conv(h2_in, out_dim, **({"heads": 1, "concat": False} if conv is GATConv else {}))
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, 0.5, training=self.training)
        return self.c2(x, ei)

def run_gnn(name, conv, **kw):
    model = GNN(conv, data.num_node_features, **kw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weight)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(1); scores = out.softmax(1)[:, 1]
    return evaluate(pred, scores, data.test_mask)

def run_sklearn(name, clf):
    Xtr = data.x[data.train_mask].cpu().numpy(); ytr = data.y[data.train_mask].cpu().numpy()
    Xte = data.x.cpu().numpy(); 
    clf.fit(Xtr, ytr)
    pred = torch.tensor(clf.predict(Xte))
    scores = torch.tensor(clf.predict_proba(Xte)[:, 1])
    return evaluate(pred, scores, data.test_mask)

results = {}
results["LogReg"]    = run_sklearn("LogReg", LogisticRegression(max_iter=1000, class_weight="balanced"))
results["RandForest"]= run_sklearn("RF", RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1))
results["GCN"]       = run_gnn("GCN", GCNConv)
results["GraphSAGE"] = run_gnn("SAGE", SAGEConv)
results["GAT"]       = run_gnn("GAT", GATConv, heads=4)

print(f"\n{'Model':<12}{'Prec':>8}{'Rec':>8}{'F1':>8}{'PR-AUC':>9}")
print("-"*45)
for k,(p,r,f,a) in results.items():
    print(f"{k:<12}{p:>8.3f}{r:>8.3f}{f:>8.3f}{a:>9.3f}")