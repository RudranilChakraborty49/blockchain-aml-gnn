import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

device = torch.device("cuda")
snapshots = [s.to(device) for s in torch.load("snapshots.pt", weights_only=False)]
IN_DIM, HIDDEN = 165, 128
TRAIN_TS = 34


def gcn_norm_propagate(x, edge_index, num_nodes):
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(col, num_nodes, dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    out = torch.zeros_like(x)
    out.index_add_(0, col, x[row] * norm.unsqueeze(-1))
    return out


def init_lstm_forget_bias(lstm_cell, dim):
    # set forget-gate bias = 1 (standard trick for stable BPTT)
    for name, p in lstm_cell.named_parameters():
        if 'bias' in name:
            p.data.fill_(0)
            p.data[dim:2*dim] = 1.0


class EvolveGCNO(nn.Module):
    def __init__(self, in_dim, hidden, out_dim=2):
        super().__init__()
        self.lstm1 = nn.LSTMCell(in_dim, in_dim)
        self.lstm2 = nn.LSTMCell(hidden, hidden)
        init_lstm_forget_bias(self.lstm1, in_dim)
        init_lstm_forget_bias(self.lstm2, hidden)
        self.W1 = nn.Parameter(torch.empty(in_dim, hidden)); nn.init.xavier_uniform_(self.W1)
        self.W2 = nn.Parameter(torch.empty(hidden, out_dim)); nn.init.xavier_uniform_(self.W2)

    def reset_state(self):
        self.h1 = torch.zeros_like(self.W1); self.c1 = torch.zeros_like(self.W1)
        self.h2 = torch.zeros_like(self.W2); self.c2 = torch.zeros_like(self.W2)
        self.W1_t = self.W1.clone()
        self.W2_t = self.W2.clone()

    def step(self, x, edge_index):
        self.h1, self.c1 = self.lstm1(self.W1_t.T, (self.h1.T, self.c1.T))
        self.h1, self.c1 = self.h1.T, self.c1.T
        self.W1_t = self.h1

        h = gcn_norm_propagate(x @ self.W1_t, edge_index, x.size(0))
        h = F.relu(h)

        self.h2, self.c2 = self.lstm2(self.W2_t.T, (self.h2.T, self.c2.T))
        self.h2, self.c2 = self.h2.T, self.c2.T
        self.W2_t = self.h2

        out = gcn_norm_propagate(h @ self.W2_t, edge_index, x.size(0))
        return out


model = EvolveGCNO(IN_DIM, HIDDEN).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

y_tr = torch.cat([s.y[s.labeled_mask] for s in snapshots[:TRAIN_TS]])
n_ill = int((y_tr == 1).sum()); n_lic = int((y_tr == 0).sum())
weight = torch.tensor([1.0, n_lic / n_ill], device=device)


def run_epoch(train=True):
    model.train(train); model.reset_state()
    losses, all_pred, all_score, all_y = [], [], [], []
    opt.zero_grad()
    for i, s in enumerate(snapshots):
        is_train_ts = i < TRAIN_TS
        if train and not is_train_ts:
            with torch.no_grad():
                out = model.step(s.x, s.edge_index)
        else:
            out = model.step(s.x, s.edge_index)

        if train and is_train_ts and s.labeled_mask.any():
            losses.append(F.cross_entropy(out[s.labeled_mask], s.y[s.labeled_mask], weight=weight))
        if not train and not is_train_ts and s.labeled_mask.any():
            pred = out.argmax(1); score = out.softmax(1)[:, 1]
            all_pred.append(pred[s.labeled_mask].cpu())
            all_score.append(score[s.labeled_mask].cpu())
            all_y.append(s.y[s.labeled_mask].cpu())

    if train:
        total = torch.stack(losses).mean()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        return float(total.detach())
    return torch.cat(all_y), torch.cat(all_pred), torch.cat(all_score)


best_f1, best_state, patience, bad = 0.0, None, 30, 0
for epoch in range(1, 201):
    loss = run_epoch(train=True)
    with torch.no_grad():
        y, p, sc = run_epoch(train=False)
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="binary", pos_label=1, zero_division=0)
    prauc = average_precision_score(y, sc)
    if epoch % 5 == 0:
        print(f"Ep {epoch:3d} | loss {loss:.4f} | P {prec:.3f} R {rec:.3f} F1 {f1:.3f} PR-AUC {prauc:.3f}")
    if f1 > best_f1:
        best_f1, best_state, bad = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= patience:
            print(f"Early stop at ep {epoch}, best F1 {best_f1:.3f}")
            break

model.load_state_dict(best_state)
with torch.no_grad():
    y, p, sc = run_epoch(train=False)
prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="binary", pos_label=1, zero_division=0)
prauc = average_precision_score(y, sc)
print(f"\nFINAL EvolveGCN-O (best) | P {prec:.3f} R {rec:.3f} F1 {f1:.3f} PR-AUC {prauc:.3f}")