import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

device = torch.device("cuda")
data = EllipticBitcoinDataset(root="data/elliptic")[0].to(device)

class DOMINANT(nn.Module):
    def __init__(self, in_dim, hidden=64, emb=32):
        super().__init__()
        self.enc1 = GCNConv(in_dim, hidden)
        self.enc2 = GCNConv(hidden, emb)
        self.dec1 = GCNConv(emb, hidden)
        self.dec2 = GCNConv(hidden, in_dim)
    def forward(self, x, ei):
        h = F.relu(self.enc1(x, ei))
        z = self.enc2(h, ei)
        h2 = F.relu(self.dec1(z, ei))
        x_hat = self.dec2(h2, ei)
        return x_hat, z

model = DOMINANT(data.num_node_features).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

ALPHA = 0.8

def loss_fn(x, x_hat, z, pos_ei):
    feat_err = ((x - x_hat) ** 2).sum(dim=1)
    feat_loss = feat_err.mean()
    neg_ei = negative_sampling(pos_ei, num_nodes=x.size(0), num_neg_samples=pos_ei.size(1))
    pos_score = (z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)
    neg_score = (z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)
    struct_loss = F.binary_cross_entropy_with_logits(
        torch.cat([pos_score, neg_score]),
        torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    )
    return ALPHA * feat_loss + (1 - ALPHA) * struct_loss, feat_err

def anomaly_scores(x, x_hat, z, ei):
    feat_err = ((x - x_hat) ** 2).sum(dim=1)
    pos_score = torch.sigmoid((z[ei[0]] * z[ei[1]]).sum(dim=1))
    struct_err = torch.zeros(x.size(0), device=x.device)
    cnt = torch.zeros(x.size(0), device=x.device)
    struct_err.index_add_(0, ei[0], 1 - pos_score)
    cnt.index_add_(0, ei[0], torch.ones_like(pos_score))
    struct_err = struct_err / cnt.clamp(min=1)
    return ALPHA * feat_err + (1 - ALPHA) * struct_err

for epoch in range(1, 201):
    model.train(); opt.zero_grad()
    x_hat, z = model(data.x, data.edge_index)
    loss, _ = loss_fn(data.x, x_hat, z, data.edge_index)
    loss.backward(); opt.step()
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            x_hat, z = model(data.x, data.edge_index)
            scores = anomaly_scores(data.x, x_hat, z, data.edge_index)
        y = data.y[data.test_mask].cpu(); s = scores[data.test_mask].cpu()
        roc = roc_auc_score(y, s); prauc = average_precision_score(y, s)
        print(f"Ep {epoch:3d} | loss {float(loss):.4f} | ROC-AUC {roc:.3f} | PR-AUC {prauc:.3f}")

model.eval()
with torch.no_grad():
    x_hat, z = model(data.x, data.edge_index)
    scores = anomaly_scores(data.x, x_hat, z, data.edge_index)
y = data.y[data.test_mask].cpu(); s = -scores[data.test_mask].cpu()
roc = roc_auc_score(y, s); prauc = average_precision_score(y, s)
print(f"\nFINAL DOMINANT (unsupervised) | ROC-AUC {roc:.3f} | PR-AUC {prauc:.3f}")