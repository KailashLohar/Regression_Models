import streamlit as st
import pandas as pd
from models.GNN.gnn_predict import predict_gnn
from models.Random_Forest.rf_predict import predict_rf
from utils import find_smiles_column
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from custom_styles import load_custom_styles

st.set_page_config(page_title="Molecular Property Prediction using GNN and RF", layout="wide")

st.markdown(load_custom_styles(), unsafe_allow_html=True)

st.markdown("""
    <div class="title-box">
        <h1>Molecular Property Prediction</h1>
    </div>
    <h4 style='text-align: center; color: grey;'>using Graph Neural Network and Random Forest</h4>
    """, unsafe_allow_html=True
)


st.sidebar.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        background-color: #80deea;
    }
    .sidebar .sidebar-content {
        margin-top: -50px;
    }
    </style>
    """, unsafe_allow_html=True
)
st.sidebar.image("logo.svg", width=200)

st.sidebar.title("Input Data")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
single_smiles = st.sidebar.text_input("Or enter a single SMILES string")
submit_button = st.sidebar.button("Predict")

import base64
from io import BytesIO

def display_results_with_structures(results):
    smiles_list = results['SMILES'].tolist()
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    
    # Generate images with higher DPI for better clarity
    imgs = [Draw.MolToImage(mol, size=(200, 100), kekulize=True, dpi=1200) for mol in mols]
    
    # Convert images to base64
    img_strs = []
    for img in imgs:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        img_strs.append(img_str)
    
    results['Structure'] = img_strs
    try:
        results_display = results[['SMILES', 'Structure', 'GNN_Prediction', 'RF_Prediction']]
    except KeyError as e:
        st.error(f"KeyError: {e}")
        st.write("Columns in results dataframe:", results.columns)
        return
    
    st.markdown("<span style='color:#880e4f; font-weight:bold;'>Predictions with Molecular Structures:</span>", unsafe_allow_html=True)
    
    # Add custom CSS for the zoom effect
    st.markdown(
        """
        <style>
        .image-container {
            display: inline-block;
            position: relative;
        }
        .zoom-img {
            transition: transform 0.3s ease; /* Animation */
            cursor: zoom-in;
        }
        .zoom-img:hover {
            transform: scale(2.5); /* Zoom factor */
            z-index: 1000; /* Ensure the image appears above other elements */
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    for i in range(0, len(results_display), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(results_display):
                with cols[j]:
                    row = results_display.iloc[i + j]
                    st.write(f"**SMILES:** {row['SMILES']}")
                    
                    st.markdown(f"""
                    <div class="image-container">
                        <img class="zoom-img" src="data:image/png;base64,{row['Structure']}"/>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"<p style='margin-bottom:0'><strong>GNN Prediction:</strong> {row['GNN_Prediction']:.2f}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-bottom:0'><strong>RF Prediction:</strong> {row['RF_Prediction']:.2f}</p>", unsafe_allow_html=True)
                    st.markdown("<hr style='margin-top:5px; margin-bottom:5px;'>", unsafe_allow_html=True)



if submit_button:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV file preview:")
        st.write(df.head())
        smiles_col = find_smiles_column(df)
        if smiles_col:
            df = df.rename(columns={smiles_col: 'SMILES'})
            true_values = df['Target'] if 'Target' in df.columns else None
            df = df.drop(columns=['Target'], errors='ignore')

            gnn_results = predict_gnn(df)
            gnn_results = gnn_results.rename(columns={'Prediction': 'GNN_Prediction'})
            rf_results = predict_rf(df)
            rf_results = rf_results.rename(columns={'Prediction': 'RF_Prediction'})
            results = pd.merge(gnn_results, rf_results, on='SMILES')

            st.markdown("<span style='color:#880e4f; font-weight:bold;'>Regression Predictions using GNN and RF Model::</span>", unsafe_allow_html=True)
            st.write(results)

            display_results_with_structures(results)

            cols = st.columns(2)
            
            with cols[0]:
                with st.expander("GNN Model Information"):
                    st.write("• RMSE value of GNN Model: 20.9")
                    st.write("• Predictions are based on 5 models obtained using 5-fold cross-validation.")
                    st.image("models/GNN/model_files/actual_vs_predicted.png", caption="Actual vs Predicted (GNN)")
                    st.image("models/GNN/model_files/mu_sigma.png", caption="Mu Sigma (GNN)")
                    st.image("models/GNN/model_files/tsne_train_vs_test_data.png", caption="t-SNE Train vs Test (GNN)")
            
            with cols[1]:
                with st.expander("RF Model Information"):
                    st.write("• RMSE value of RF Model: 23.39")
                    st.write("• Predictions are based on the model with best parameters obtained using Grid Search CV.")
                    st.image("models/Random_Forest/model_files/actual_vs_predicted.png", caption="Actual vs Predicted (RF)")
                    st.image("models/Random_Forest/model_files/mu_sigma.png", caption="Mu Sigma (RF)")
                    st.image("models/Random_Forest/model_files/tsne_train_vs_test_data.png", caption="t-SNE Train vs Test (RF)")
        else:
            st.error("The CSV file must contain a column with SMILES strings.")
    elif single_smiles:
        st.markdown("<span style='color:#512da8; font-weight:bold;'>Entered SMILES string:</span>", unsafe_allow_html=True)
        st.write(single_smiles)
        df = pd.DataFrame({'SMILES': [single_smiles]})
        gnn_results = predict_gnn(df)
        gnn_results = gnn_results.rename(columns={'Prediction': 'GNN_Prediction'})
        rf_results = predict_rf(df)
        rf_results = rf_results.rename(columns={'Prediction': 'RF_Prediction'})
        results = pd.merge(gnn_results, rf_results, on='SMILES')

        st.markdown("<span style='color:#880e4f; font-weight:bold;'>Regression Predictions using GNN and RF Model:</span>", unsafe_allow_html=True)
        st.write(results)

        display_results_with_structures(results)

        cols = st.columns(2)
        
        with cols[0]:
            with st.expander("GNN Model Information"):
                st.write("• RMSE value of GNN Model: 20.9")
                st.write("• Predictions are based on 5 models obtained using 5-fold cross-validation.")
                st.image("models/GNN/model_files/actual_vs_predicted.png", caption="Actual vs Predicted (GNN)")
                st.image("models/GNN/model_files/mu_sigma.png", caption="Mu Sigma (GNN)")
                st.image("models/GNN/model_files/tsne_train_vs_test_data.png", caption="t-SNE Train vs Test (GNN)")
        
        with cols[1]:
            with st.expander("RF Model Information"):
                st.write("• RMSE value of RF Model: 23.39")
                st.write("• Predictions are based on the model with best parameters obtained using Grid Search CV.")
                st.image("models/Random_Forest/model_files/actual_vs_predicted.png", caption="Actual vs Predicted (RF)")
                st.image("models/Random_Forest/model_files/mu_sigma.png", caption="Mu Sigma (RF)")
                st.image("models/Random_Forest/model_files/tsne_train_vs_test_data.png", caption="t-SNE Train vs Test (RF)")
    else:
        st.error("Please upload a CSV file or enter a SMILES string for prediction.")
