"""Browse the molecular dataset hub and train on one dataset with its source split.

"""
from torch_molecule import GNNMolecularPredictor
from torch_molecule.datasets import list_datasets, load_dataset

for row in list_datasets(category="adme", grade="ready"):
    print(f"{row['name']:45} {row['access']:7} {row['license_class']:12} {row['application']}")

name = "moleculeace_chembl204_ki"
train = load_dataset(name, split="train")
test = load_dataset(name, split="test")
print(train.info["targets"], [tag["label"] for tag in train.info["quality"]["tags"]])

model = GNNMolecularPredictor(num_task=train.target.shape[1], task_type="regression", epochs=5)
model.fit(train.data, train.target)
print(model.predict(test.data)["prediction"][:5])
