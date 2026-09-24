from rdkit.Chem import QED

# This is used as the default reward function for MolGAN
def qed_reward_fn(mol):
    """
    Computes the QED score of a single RDKit Mol object.
    Returns 0.0 for invalid molecules or errors.
    """
    if mol is not None:
        try:
            return QED.qed(mol)
        except Exception:
            return 0.0
    return 0.0
