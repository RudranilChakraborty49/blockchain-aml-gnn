from torch_geometric.datasets import EllipticBitcoinDataset

ds = EllipticBitcoinDataset(root="data/elliptic")
data = ds[0]
print(data)
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
print("Features:", data.num_node_features)
print("Classes:", ds.num_classes)