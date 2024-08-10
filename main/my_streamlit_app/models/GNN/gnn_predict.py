import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F 
from torch_geometric.loader import DataLoader  
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from ogb.utils import smiles2graph

torch.manual_seed(42)

class MolecularGraphNeuralNetwork(torch.nn.Module):
    def __init__(self, dropout_rate, leaky_relu_alpha, embedding_size):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.initial_conv = GCNConv(9, embedding_size)
        self.conv1 = SAGEConv(embedding_size, embedding_size)
        self.conv2 = GraphConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.fc1 = Linear(embedding_size * 3, embedding_size)
        self.fc2 = Linear(embedding_size, embedding_size)  
        self.fc3 = Linear(embedding_size, 1)
        self.bn1 = BatchNorm1d(embedding_size)
        self.bn2 = BatchNorm1d(embedding_size)
        self.bn3 = BatchNorm1d(embedding_size)  
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)  
        self.leaky_relu_alpha = leaky_relu_alpha

    def forward(self, x, edge_index, batch_index):
        x = self.initial_conv(x, edge_index)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_alpha)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_alpha)
        x = self.bn2(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_alpha)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_alpha)
        x = torch.cat([global_max_pool(x, batch_index), global_mean_pool(x, batch_index), global_add_pool(x, batch_index)], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_alpha)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_alpha)
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        return x

class CustomMoleculeNetDataset_predict(InMemoryDataset):
    def __init__(self, data_list):
        super(CustomMoleculeNetDataset_predict, self).__init__(".", transform=None, pre_transform=None)
        self.data_list = data_list
        self.data, self.slices = self.collate(data_list)

    @staticmethod
    def create_data_list(df):
        data_list = []
        for _, row in df.iterrows():
            graph = smiles2graph(row['SMILES'])
            data = Data(
                x=torch.tensor(graph['node_feat']),
                edge_index=torch.tensor(graph['edge_index']),
                edge_attr=torch.tensor(graph['edge_feat'])
            )
            data.smiles = row['SMILES']
            data_list.append(data)
        return data_list

def predict_gnn(df):
    NUM_FOLDS = 5
    num_graphs_per_batch = 16
    best_dropout_rate, best_leaky_relu_alpha, best_embedding_size = 0.05, 0.1, 512

    test_data = CustomMoleculeNetDataset_predict.create_data_list(df)
    test_loader = DataLoader(test_data, batch_size=num_graphs_per_batch)

    models = []
    for fold in range(NUM_FOLDS):
        model = MolecularGraphNeuralNetwork(best_dropout_rate, best_leaky_relu_alpha, best_embedding_size)
        model_checkpoint_path = os.path.join('models', 'GNN', 'model_files', f'model_fold_{fold+1}_dropout_{best_dropout_rate}_alpha_{best_leaky_relu_alpha}_embed_{best_embedding_size}.pth')
        checkpoint = torch.load(model_checkpoint_path, map_location=torch.device('cpu')) 

        if 'module.' in list(checkpoint.keys())[0]:
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        model.load_state_dict(checkpoint)  
        model.eval()
        models.append(model)

    predictions = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch in test_loader:
        batch = batch.to(device)
        batch_predictions = []
        for model in models:
            model = model.to(device)  
            with torch.no_grad():
                pred = model(batch.x.float().to(device), batch.edge_index.to(device), batch.batch.to(device))
                batch_predictions.append(pred.cpu().numpy())

        batch_predictions = np.concatenate(batch_predictions, axis=1)
        mean_predictions = batch_predictions.mean(axis=1)
        predictions.extend(mean_predictions)

    test_results = pd.DataFrame({'SMILES': df['SMILES'], 'GNN_Prediction': predictions})
    return test_results
