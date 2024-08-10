# utils.py

import pandas as pd

def find_smiles_column(df):
    possible_columns = ['smiles', 'smile', 'compounds', 'molecule']
    for col in df.columns:
        if any(possible in col.lower() for possible in possible_columns):
            return col
    return None
