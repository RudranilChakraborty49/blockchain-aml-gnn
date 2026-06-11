import torch
import torch.nn.functional as F
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import GCNConv
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = EllipticBitcoinDataset(root="data/elliptic")[0].to(device)

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

model = GCN(data.num_node_features).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# class imbalance: weight illicit (class 1) higher
n_illicit = int(data.y[data.train_mask].sum())
n_licit = int(data.train_mask.sum()) - n_illicit
weight = torch.tensor([1.0, n_licit / n_illicit], device=device)

def train():
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weight)
    loss.backward(); opt.step()
    return float(loss.detach())

@torch.no_grad()
def test(mask):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    scores = model(data.x, data.edge_index).softmax(dim=1)[:, 1]
    y = data.y[mask].cpu(); p = pred[mask].cpu()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, p, average="binary", pos_label=1, zero_division=0)
    pr_auc = average_precision_score(y, scores[mask].cpu())
    return prec, rec, f1, pr_auc

for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        prec, rec, f1, pr_auc = test(data.test_mask)
        print(f"Ep {epoch:3d} | loss {loss:.4f} | "
              f"P {prec:.3f} R {rec:.3f} F1 {f1:.3f} PR-AUC {pr_auc:.3f}")

prec, rec, f1, pr_auc = test(data.test_mask)
print(f"\nFINAL GCN (temporal) | P {prec:.3f} R {rec:.3f} F1 {f1:.3f} PR-AUC {pr_auc:.3f}")