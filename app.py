import torch, torch.nn as nn, torch.nn.functional as F
import gradio as gr
import json, os, pandas as pd
from pyvis.network import Network
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph, to_undirected

# -------- model --------
class SAGE(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=2):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, out_dim)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        return self.c2(x, ei)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading dataset...")
data = EllipticBitcoinDataset(root="data/elliptic")[0].to(device)
edge_index_undirected = to_undirected(data.edge_index)

print("Loading model...")
model = SAGE(data.num_node_features).to(device)
model.load_state_dict(torch.load("models/elliptic1_sage.pt", map_location=device))
model.eval()

print("Running model on full graph...")
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    probs = logits.softmax(1)[:, 1].cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()

raw_dir = os.path.join("data/elliptic", "raw")
feats_raw = pd.read_csv(os.path.join(raw_dir, "elliptic_txs_features.csv"), header=None, usecols=[0])
txid_list = feats_raw[0].values
txid_to_idx = {tx: i for i, tx in enumerate(txid_list)}

with open("results/cross_dataset_results.json") as f:
    xd = json.load(f)
print(f"Ready. {len(txid_list):,} transactions indexed. Cross-dataset results loaded.")

LABEL = {0: "licit", 1: "illicit", -1: "unknown"}
def y_of(i):
    yv = int(data.y[i].cpu())
    return yv if yv in (0, 1) else -1
COLOR = {"illicit": "#e24b4a", "licit": "#1d9e75", "unknown": "#888780"}

# ============================================================
# TAB 1: Transaction Lookup
# ============================================================
def analyze(tx_input, hops):
    try:
        tx = int(tx_input)
    except (ValueError, TypeError):
        return "Invalid transaction ID.", "<p>No graph.</p>"
    if tx not in txid_to_idx:
        return f"Transaction ID {tx} not found.", "<p>No graph.</p>"

    center = txid_to_idx[tx]
    subset, sub_ei, _, _ = k_hop_subgraph(
        center, hops, edge_index_undirected, relabel_nodes=True, num_nodes=data.num_nodes)
    sub_nodes = subset.cpu().numpy()
    sub_ei_np = sub_ei.cpu().numpy()

    p_illicit = float(probs[center])
    pred_label = "ILLICIT" if preds[center] == 1 else "LICIT"
    true_label = LABEL[y_of(center)]

    net = Network(height="600px", width="100%", bgcolor="#ffffff",
                  font_color="#000000", directed=True, cdn_resources="in_line")
    net.toggle_physics(True)
    for local_i, global_i in enumerate(sub_nodes):
        true_lbl = LABEL[y_of(global_i)]
        pred = "illicit" if preds[global_i] == 1 else "licit"
        p = float(probs[global_i])
        is_center = (global_i == center)
        title = (f"txId: {txid_list[global_i]}\n"
                 f"Predicted: {pred} ({p:.1%} illicit)\n"
                 f"True label: {true_lbl}")
        size = 35 if is_center else (15 + p * 20)
        net.add_node(int(local_i), label=str(txid_list[global_i])[:8],
                     title=title, color=COLOR[pred], size=size,
                     borderWidth=4 if is_center else 1)
    for s, d in zip(sub_ei_np[0], sub_ei_np[1]):
        net.add_edge(int(s), int(d), color="#cccccc", width=0.5)

    raw_html = net.generate_html(notebook=False)
    escaped = raw_html.replace('&', '&amp;').replace('"', '&quot;')
    iframe_html = f'<iframe srcdoc="{escaped}" width="100%" height="620" style="border:none;border-radius:8px;"></iframe>'

    summary = (
        f"### Transaction `{tx}`\n\n"
        f"- **Predicted:** {pred_label} ({p_illicit:.1%} illicit probability)\n"
        f"- **Ground truth:** {true_label}\n"
        f"- **Neighborhood:** {len(sub_nodes)} nodes, {sub_ei_np.shape[1]} edges within {hops} hop(s)\n"
        f"- **Illicit nodes nearby:** {int((preds[sub_nodes] == 1).sum())}\n\n"
        f"Red = predicted illicit. Green = licit. Hover for details."
    )
    return summary, iframe_html

# ============================================================
# TAB 2: Cross-Dataset Failure
# ============================================================
def cross_dataset_view():
    e = xd["elliptic_to_elliptic"]; a = xd["amlw_to_amlw"]
    ea = xd["elliptic_to_amlw"]; ae = xd["amlw_to_elliptic"]
    e_drop = (e["F1"] - ea["F1"]) / e["F1"] * 100
    a_drop = (a["F1"] - ae["F1"]) / a["F1"] * 100

    summary = f"""### The Cross-Dataset Failure

We trained the same Graph Neural Network (GraphSAGE) on two different Bitcoin AML datasets — **Elliptic** (real Bitcoin transactions, 200k nodes) and **AMLWorld** (IBM synthetic banking transactions, 200k nodes) — using the same 11 graph-structural features for both. Then we cross-evaluated: train on one, test on the other.

**Result: models do not transfer between AML datasets.**

| Train → Test | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Elliptic → Elliptic *(in-domain)* | **{e['F1']:.3f}** | {e['PR_AUC']:.3f} | {e['ROC']:.3f} |
| AMLWorld → AMLWorld *(in-domain)* | **{a['F1']:.3f}** | {a['PR_AUC']:.3f} | {a['ROC']:.3f} |
| Elliptic → AMLWorld *(cross)* | **{ea['F1']:.3f}** | {ea['PR_AUC']:.3f} | {ea['ROC']:.3f} |
| AMLWorld → Elliptic *(cross)* | **{ae['F1']:.3f}** | {ae['PR_AUC']:.3f} | {ae['ROC']:.3f} |

**Key findings:**
- Elliptic → AMLWorld F1 drops **{e_drop:.0f}%** (from {e['F1']:.3f} to {ea['F1']:.3f})
- AMLWorld → Elliptic F1 drops **{a_drop:.0f}%** (from {a['F1']:.3f} to {ae['F1']:.3f})
- **Elliptic → AMLWorld ROC is {ea['ROC']:.2f} — *worse than random*.** The Elliptic-trained model is actively anti-predictive on AMLWorld.

**Why this matters.** AML research uses dataset-specific models without testing transfer. Our finding shows that "F1 = 0.66 on Elliptic" tells you nothing about real-world deployment, where new transaction patterns will differ from training data. The field needs cross-dataset evaluation as standard practice.
"""

    bars = [
        ("Elliptic → Elliptic", e["F1"], "#1d9e75"),
        ("AMLWorld → AMLWorld", a["F1"], "#1d9e75"),
        ("Elliptic → AMLWorld", ea["F1"], "#e24b4a"),
        ("AMLWorld → Elliptic", ae["F1"], "#e24b4a"),
    ]
    max_f1 = max(b[1] for b in bars)
    chart_rows = ""
    for label, val, color in bars:
        width_pct = (val / max_f1) * 100
        chart_rows += f"""
        <div style="margin: 8px 0;">
          <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 220px; font-size: 13px; color: #222; font-weight: 500;">{label}</div>
            <div style="flex: 1; background: #f0f0f0; border-radius: 4px; height: 28px; position: relative;">
              <div style="width: {width_pct}%; background: {color}; height: 100%; border-radius: 4px;"></div>
              <div style="position: absolute; left: 8px; top: 4px; color: #fff; font-weight: 500; font-size: 13px;">F1 = {val:.3f}</div>
            </div>
          </div>
        </div>
        """
    chart_html = f"""
    <div style="background: #fff; padding: 20px; border-radius: 8px; color: #222;">
      <h3 style="margin-top: 0; color: #222;">F1 Score Comparison</h3>
      <p style="color: #555; font-size: 13px;">Green = in-domain, red = cross-dataset</p>
      {chart_rows}
    </div>
    """
    return summary, chart_html

# ============================================================
# UI
# ============================================================
known_illicit = [int(txid_list[i]) for i in range(len(txid_list))
                 if y_of(i) == 1 and preds[i] == 1][:3]
known_licit = [int(txid_list[i]) for i in range(len(txid_list))
               if y_of(i) == 0][:2]
example_ids = known_illicit + known_licit

with gr.Blocks(title="Bitcoin AML Detection") as app:
    gr.Markdown("# Blockchain Transaction Anomaly Detection\nA Graph Neural Network approach to flagging illicit Bitcoin transactions, with a cross-dataset robustness study.")
    with gr.Tabs():
        with gr.Tab("Transaction Lookup"):
            gr.Markdown("Enter a Bitcoin transaction ID. The model classifies it and shows its neighborhood graph. Red = illicit, green = licit.")
            with gr.Row():
                tx_in = gr.Textbox(label="Transaction ID", value=str(example_ids[0]))
                hops = gr.Slider(1, 3, value=2, step=1, label="Neighborhood hops")
                btn = gr.Button("Analyze", variant="primary")
            summary_out = gr.Markdown()
            graph_out = gr.HTML()
            btn.click(analyze, inputs=[tx_in, hops], outputs=[summary_out, graph_out])
            gr.Examples([[str(t), 2] for t in example_ids], inputs=[tx_in, hops])

        with gr.Tab("Cross-Dataset Failure"):
            xd_summary, xd_chart = cross_dataset_view()
            gr.Markdown(xd_summary)
            gr.HTML(xd_chart)

if __name__ == "__main__":
    app.launch(inbrowser=True)