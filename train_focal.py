import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

device = torch.device("cuda")
data = EllipticBitcoinDataset(root="data/elliptic")[0].to(device)

n_ill = int(data.y[data.train_mask].sum())
n_lic = int(data.train_mask.sum()) - n_ill
weight = torch.tensor([1.0, n_lic / n_ill], device=device)

class SAGE(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=2):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, out_dim)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        x = F.dropout(x, 0.5, training=self.training)
        return self.c2(x, ei)

def focal_loss(logits, y, alpha=0.75, gamma=2.0):
    """Binary focal loss for 2-class logits."""
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    # alpha_t: alpha for class 1, (1-alpha) for class 0
    alpha_t = torch.where(y == 1, alpha, 1 - alpha)
    p_t = p.gather(1, y.unsqueeze(1)).squeeze(1)
    log_p_t = log_p.gather(1, y.unsqueeze(1)).squeeze(1)
    loss = -alpha_t * (1 - p_t) ** gamma * log_p_t
    return loss.mean()

def evaluate(out):
    pred = out.argmax(1); scores = out.softmax(1)[:, 1]
    y = data.y[data.test_mask].cpu(); p = pred[data.test_mask].cpu()
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="binary", pos_label=1, zero_division=0)
    prauc = average_precision_score(y, scores[data.test_mask].cpu())
    return prec, rec, f1, prauc

def train_with(loss_name):
    torch.manual_seed(42)
    model = SAGE(data.num_node_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        if loss_name == "weighted_ce":
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weight)
        else:
            loss = focal_loss(out[data.train_mask], data.y[data.train_mask], alpha=0.75, gamma=2.0)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    return evaluate(out)

print(f"{'Loss':<15}{'Prec':>8}{'Rec':>8}{'F1':>8}{'PR-AUC':>9}")
print("-"*48)
for name in ["weighted_ce", "focal"]:
    p, r, f, a = train_with(name)
    print(f"{name:<15}{p:>8.3f}{r:>8.3f}{f:>8.3f}{a:>9.3f}")