import pandas as pd
import numpy as np
from torch_molecule import SupervisedMolecularEncoder, AttrMaskMolecularEncoder, ContextPredMolecularEncoder, EdgePredMolecularEncoder, MoamaMolecularEncoder
import os

# Load training data
path_to_data = ''
train_data = pd.read_csv(f'{path_to_data}/train.csv')