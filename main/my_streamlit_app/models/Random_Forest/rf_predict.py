import os
import pandas as pd
from joblib import load
from rdkit import Chem
from rdkit.Chem import AllChem

def calculate_ecfps(df, smiles_column='SMILES', radius=2, n_bits=1024):
    def get_mol_ecfp(mol):
        if mol:
            return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
        else:
            return [0] * n_bits
    
    mols = [Chem.MolFromSmiles(x) for x in df[smiles_column].values.tolist()]
    ecfps = [get_mol_ecfp(mol) for mol in mols]
    df_ecfp = pd.DataFrame(ecfps)
    return df_ecfp

def predict_rf(df):
    model_path = os.path.join('models', 'Random_Forest', 'model_files', 'rf_model.joblib')
    rf_model = load(model_path)
    
    training_columns_path = os.path.join('models', 'Random_Forest', 'model_files', 'training_columns.csv')
    training_columns = pd.read_csv(training_columns_path).squeeze().tolist()
    
    df_ecfp = calculate_ecfps(df)
    df_ecfp = df_ecfp.reindex(columns=training_columns, fill_value=0)
    expected_features = rf_model.n_features_in_
    if df_ecfp.shape[1] != expected_features:
        raise ValueError(f"X has {df_ecfp.shape[1]} features, but RandomForestRegressor is expecting {expected_features} features as input.")
    predictions = rf_model.predict(df_ecfp)
    results = pd.DataFrame({'SMILES': df['SMILES'], 'RF_Prediction': predictions})
    return results
