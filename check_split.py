from torch_geometric.datasets import EllipticBitcoinDataset

data = EllipticBitcoinDataset(root="data/elliptic")[0]
print("Train labeled nodes:", int(data.train_mask.sum()))
print("Test labeled nodes :", int(data.test_mask.sum()))
print("Illicit in train:", int(data.y[data.train_mask].sum()))
print("Illicit in test :", int(data.y[data.test_mask].sum()))
print("Illicit rate train: %.3f" % (data.y[data.train_mask].float().mean()))
print("Illicit rate test : %.3f" % (data.y[data.test_mask].float().mean()))