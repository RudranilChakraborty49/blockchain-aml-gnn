# save_elliptic1_sage.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import SAGEConv

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

model = SAGE(data.num_node_features).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for ep in range(200):
    model.train(); opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weight)
    loss.backward(); opt.step()
import os; os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/elliptic1_sage.pt")
print("saved models/elliptic1_sage.pt")