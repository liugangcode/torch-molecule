from torch_molecule import GREAMolecularPredictor
X_train = ['C', 'CC', ...]
y_train = [1.0, 2.0, ...]

predictor = GREAMolecularPredictor(num_tasks=1, task_type="regression")
# auto hyperparameter search for the model
predictor.autofit(X_train, y_train, n_trials=100)
# prediction
X_test = ['CC', 'CCO', ...]
output = predictor.predict(X_test)
y_test, y_var = output['prediction'], output['variance']


from torch_molecule import GraphDITMolecularGenerator
X_train = ['C', 'CC', ...]
y_train = [1.0, 2.0, ...]

generator = GraphDITMolecularGenerator(y_dim=1, task_type=['regression'])
generator.fit(X_train, y_train)
# conditional generation
y_test = [0.5, 1.5, ...]
smiles_list = generator.generate(y_test)


## encoder
from torch_molecule import AttrMaskMolecularEncoder
X_train = ['C', 'CC', ...]

encoder = AttrMaskMolecularEncoder()
encoder.fit(X_train)
# encoding
X_test = ['CC', 'CCO', ...]
encoder.encode(X_test)
