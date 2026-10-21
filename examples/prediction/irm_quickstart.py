import numpy as np

from torch_molecule import IRMMolecularPredictor

smiles = ["CCO", "CCCO", "c1ccccc1", "Cc1ccccc1"]
targets = np.array([[0.0], [0.3], [1.6], [1.8]])  # Illustrative property values.

model = IRMMolecularPredictor(
    task_type="regression",
    IRM_environment="random",
    num_layer=2,
    hidden_size=32,
    batch_size=4,
    epochs=5,
    device="cpu",
)
model.fit(smiles, targets)
predictions = model.predict(["CCCO", "CCC"])["prediction"]
print(predictions.shape)  # (2, 1)
