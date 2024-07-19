import torch
import sys
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch_geometric.data import DataLoader
from gnn_tr import GNN_prediction, TR
import torch.nn.functional as F
from data import MoleculeDataset, random_sampler_test
import torch.optim as optim
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, balanced_accuracy_score
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def load_ckp(checkpoint_fpath, model, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
    checkpoint = torch.load(checkpoint_fpath, map_location = device)
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.to(device)
      
    optimizer_to(optimizer, device)

    return model, optimizer, checkpoint['epoch']

def metrics(roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores, y_label, y_pred):
    
    roc_auc_list = []
    f1_score_list = []
    precision_score_list = []
    sn_score_list = []
    sp_score_list = []
    acc_score_list = []
    bacc_score_list = []

    y_label = torch.cat(y_label, dim = 0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cpu().detach().numpy()
   
    roc_auc_list.append(roc_auc_score(y_label, y_pred))
    roc_auc = sum(roc_auc_list)/len(roc_auc_list)

    f1_score_list.append(f1_score(y_label, y_pred))
    f1_scr = sum(f1_score_list)/len(f1_score_list)

    precision_score_list.append(precision_score(y_label, y_pred))
    p_scr = sum(precision_score_list)/len(precision_score_list)

    sn_score_list.append(recall_score(y_label, y_pred))
    sn_scr = sum(sn_score_list)/len(sn_score_list)

    tn, fp, fn, tp = confusion_matrix(y_label, y_pred).ravel()
    sp_score_list.append(tn/(tn+fp))
    sp_scr = sum(sp_score_list)/len(sp_score_list)

    acc_score_list.append(accuracy_score(y_label, y_pred))
    acc_scr =  sum(acc_score_list)/len(acc_score_list)

    bacc_score_list.append(balanced_accuracy_score(y_label, y_pred))
    bacc_scr =  sum(bacc_score_list)/len(bacc_score_list)

    roc_scores = roc_auc
    f1_scores = f1_scr
    p_scores = p_scr
    sn_scores = sn_scr
    sp_scores = sp_scr
    acc_scores = acc_scr
    bacc_scores = bacc_scr

    return roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores


def sample_test(tasks, test_task, data, batch_size, n_support, n_query, exp):
    
    train = False
    dataset = MoleculeDataset("Data/" + data + "/pre-processed/task_" + str(tasks-test_task), dataset = data)
    dataset_test = MoleculeDataset("Data/" + "val" + "/pre-processed/task_1", dataset = data)
    support_dataset, query_dataset, smiles_support, smiles_query = random_sampler_test(dataset, dataset_test, data, tasks-test_task-1, n_support, n_query, train, exp+1)
    support_set = DataLoader(support_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last=True)
    smiles_ss = DataLoader(smiles_support, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last=True)
    query_set = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last = True)
    smiles_qs= DataLoader(smiles_query, batch_size=batch_size, shuffle=False, num_workers = 0, drop_last=True)
    
    return support_set, query_set, smiles_ss, smiles_qs

    

def parse_pred(logit):
    
    pred = F.sigmoid(logit)
    pred = torch.where(pred>0.5, torch.ones_like(pred), pred)
    pred = torch.where(pred<=0.5, torch.zeros_like(pred), pred) 
    
    return pred


class GNNTR_eval(nn.Module):
    def __init__(self, dataset, gnn, support_set, pretrained, baseline):
        super(GNNTR_eval,self).__init__()
                
        if dataset == "muta":
            self.tasks = 6
            self.train_tasks  = 5
            self.test_tasks = 1
            
        self.data = dataset
        self.baseline = baseline
        self.graph_layers = 5
        self.n_support = support_set
        self.learning_rate = 0.001
        self.n_query = 128
        self.emb_size = 300
        self.batch_size = 10
        self.lr_update = 0.4
        self.k_train = 5
        self.k_test = 10
        self.device = 0
        self.gnn = GNN_prediction(self.graph_layers, self.emb_size, jk = "last", dropout_prob = 0.5, pooling = "mean", gnn_type = gnn)
        self.transformer = TR(300, (30,1), 1, 128, 5, 5, 256) 
        self.gnn.from_pretrained(pretrained)
        if self.baseline == 0:
            self.pos_weight = torch.FloatTensor([5]).to(self.device)
            self.loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            self.loss_transformer = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        if self.baseline == 1:
            self.loss = nn.BCEWithLogitsLoss()
        self.meta_opt = torch.optim.Adam(self.transformer.parameters(), lr=1e-4, weight_decay=0)

        graph_params = []
        graph_params.append({"params": self.gnn.gnn.parameters()})
        graph_params.append({"params": self.gnn.graph_pred_linear.parameters(), "lr":self.learning_rate})
        
        self.opt = optim.Adam(graph_params, lr=self.learning_rate, weight_decay=0) 
        self.gnn.to(torch.device("cuda:0"))
        
        self.opt = optim.Adam(graph_params, lr=self.learning_rate, weight_decay=0) 
        self.gnn.to(torch.device("cuda:0"))
        self.transformer.to(torch.device("cuda:0"))
        
        if self.baseline == 0:
            self.ckp_path_gnn = "checkpoints/checkpoints-GT/Meta-GTMP_GNN_muta_5.pt"
            self.ckp_path_transformer = "checkpoints/checkpoints-GT/Meta-GTMP_Transformer_muta_5.pt"
            
            #self.ckp_path_gnn = "checkpoints/checkpoints-ST/TA98/ST1-Meta-GTMP_GNN_muta_5.pt"
            #self.ckp_path_transformer = "checkpoints/checkpoints-ST/TA98/ST1-Meta-GTMP_Transformer_muta_5.pt"
            
            self.gnn, self.opt, start_epoch = load_ckp(self.ckp_path_gnn, self.gnn, self.opt)
            self.transformer, self.meta_opt, start_epoch = load_ckp(self.ckp_path_transformer, self.transformer, self.meta_opt)
                           
        elif self.baseline == 1:
            self.ckp_path_gnn = "checkpoints/checkpoints-baselines/GIN/GIN_GNN_muta_5.pt"
            
            self.gnn, self.opt, start_epoch = load_ckp(self.ckp_path_gnn, self.gnn, self.opt)

    def update_graph_params(self, loss, lr_update):
         
         grads = torch.autograd.grad(loss, self.gnn.parameters(), retain_graph=True, allow_unused=True)
         used_grads = [grad for grad in grads if grad is not None]
         
         return parameters_to_vector(used_grads), parameters_to_vector(self.gnn.parameters()) - parameters_to_vector(used_grads) * lr_update
     
    def update_tr_params(self, loss, lr_update):
        
         grads_tr = torch.autograd.grad(loss, self.transformer.parameters())
         used_grads_tr = [grad for grad in grads_tr if grad is not None]
         
         return parameters_to_vector(used_grads_tr), parameters_to_vector(self.transformer.parameters()) - parameters_to_vector(used_grads_tr) * lr_update
     
        
    def get_font_size(self, image_width, image_height):
        base_size = 12
        scale_factor = min(image_width, image_height) / 400
        return int(base_size * scale_factor)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def normalize_importances(self, importances):
        
        if not importances:
            return [0] * len(importances)

        # Apply min-max scaling
        min_val = min(importances)
        max_val = max(importances)
        
        if max_val - min_val > 0:
            scaled_importances = [(imp - min_val) / (max_val - min_val) for imp in importances]
        else:
            scaled_importances = [0.5] * len(importances)  # All importances are the same
        
        scaling_factor = 12 / np.mean(scaled_importances) 

        shifting_factor = - 6 - np.mean(scaled_importances) * scaling_factor
        
        # Apply sigmoid function to spread the values
        sigmoid_importances = [self.sigmoid(imp * scaling_factor + shifting_factor) for imp in scaled_importances]  # Adjust scaling if needed (Scaling (12), Shifting (-6))

        return sigmoid_importances
    
    def sigmoid_importances(self, importances):
        
        # Apply sigmoid function to spread the values

        importances = [self.sigmoid(imp * 12 - 6) for imp in importances] # Adjust scaling if needed (Scaling (12), Shifting (-6))
        
        return importances

    def visualize_molecule(self, smiles, node_importances, edge_importances, idx_importances, edge_idx_importances, preds, labels):
        
        for i in range(len(smiles)):
            mol = Chem.MolFromSmiles(smiles[i])
                        
            if not mol:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            
            # Initialize empty atom_colors dictionary
            atom_colors = {}
            
            # Initialize empty bond_colors dictionary
            bond_colors = {}

            # Calculate atom importances and prepare atom colors
            atom_importances = []
            for idx_node in range(len(node_importances)):
                if idx_importances[idx_node] == i:
                    atom_importances.append(node_importances[idx_node])
            
            # Calculate bond importances and prepare edge colors
            bond_importances = []
            for idx_edge in range(len(edge_importances)):
                if edge_idx_importances[idx_edge] == i:
                    bond_importances.append(edge_importances[idx_edge])
                                       
            atom_importances = self.normalize_importances(atom_importances)
            bond_importances = self.normalize_importances(bond_importances)
                                
            atom_colors = {}
            for idx in range(len(atom_importances)):
                atom_colors[idx] = (0.0, 0.0, 1.0, float(atom_importances[idx]))
                
            bond_colors = {}
            for bond in mol.GetBonds():
               idx = bond.GetIdx()
               if idx < len(bond_importances):
                   importance = bond_importances[idx]
                   bond_colors[idx] = (1.0, 0.0, 0.0, importance)
            
            drawer_mol = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
            drawer_mol.DrawMolecule(mol)
            drawer_mol.FinishDrawing()
            
            drawer_nodes = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
            opts = drawer_nodes.drawOptions()
            opts.useBWAtomPalette()
            opts.addAtomIndices = False
            drawer_nodes.SetDrawOptions(opts)
            drawer_nodes.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), highlightAtomColors=atom_colors, highlightBonds=[], highlightBondColors={})
            drawer_nodes.FinishDrawing()
            
            drawer_edges = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
            opts = drawer_edges.drawOptions()
            opts.useBWAtomPalette()
            opts.addAtomIndices = False
            drawer_edges.SetDrawOptions(opts)
            highlight_bonds = list(bond_colors.keys())
            highlight_bond_colors = {idx: bond_colors[idx] for idx in highlight_bonds}
            drawer_edges.DrawMolecule(
                mol,
                highlightAtoms=[],
                highlightBonds=highlight_bonds,
                highlightBondColors=highlight_bond_colors
            )
            drawer_edges.FinishDrawing()
                        
            drawer_nodes_edges = rdMolDraw2D.MolDraw2DCairo(1000, 1000)
            opts = drawer_nodes_edges.drawOptions()
            opts.useBWAtomPalette()
            opts.addAtomIndices = False
            opts.highlightBondWidthMultiplier = 10  # Use black-and-white atom palette
            drawer_nodes_edges.SetDrawOptions(opts)
          
            #drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), highlightAtomColors=atom_colors, highlightBonds=[], highlightBondColors={})
           
            # Highlight atoms
            highlight_atoms = list(atom_colors.keys())
            highlight_atom_colors = {idx: atom_colors[idx] for idx in highlight_atoms}
            
            # Highlight edges
            highlight_bonds = list(bond_colors.keys())
            highlight_bond_colors = {idx: bond_colors[idx] for idx in highlight_bonds}
            
            # Draw molecule with highlighted atoms and edges
            drawer_nodes_edges.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightAtomColors=highlight_atom_colors,
                highlightBonds=highlight_bonds,
                highlightBondColors=highlight_bond_colors
            )
            
            drawer_nodes_edges.FinishDrawing()
            
            png_data = drawer_nodes.GetDrawingText()
            
            # Save the initial molecule image as PNG
            with open("temp.png", "wb") as f:
                f.write(png_data)
            
            # Load the PNG image with PIL
            img_ann_nodes = Image.open("temp.png")
            draw_nodes = ImageDraw.Draw(img_ann_nodes)
            font_size = self.get_font_size(1000, 1000)
            font = ImageFont.truetype("arialbd.ttf", font_size)

            # Add text annotations for atom importances
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                if idx in atom_colors:
                    pos = drawer_nodes.GetDrawCoords(idx)
                    draw_nodes.text((pos.x + 15 , pos.y + 15), f'{atom_importances[idx]:.2f}', fill="black", font=font)
            
            png_data = drawer_edges.GetDrawingText()
            
            # Save the initial molecule image as PNG
            with open("temp2.png", "wb") as f:
                f.write(png_data)
            
            # Load the PNG image with PIL
            img_ann_edges = Image.open("temp2.png")
            draw_edges = ImageDraw.Draw(img_ann_edges)
            font_size = self.get_font_size(1000, 1000)
            font = ImageFont.truetype("arialbd.ttf", font_size)

            # Add text annotations for bond importances
            for bond in mol.GetBonds():
                idx = bond.GetIdx()
                if idx in bond_colors:
                    begin_atom = bond.GetBeginAtomIdx()
                    end_atom = bond.GetEndAtomIdx()
                    pos1 = drawer_edges.GetDrawCoords(begin_atom)
                    pos2 = drawer_edges.GetDrawCoords(end_atom)
                    pos = ((pos1.x + pos2.x) / 2, (pos1.y + pos2.y) / 2)
                    draw_edges.text((pos[0], pos[1]), f'{bond_importances[idx]:.2f}', fill="black", font=font)

            self.mol_count+=1  
            self.mol_D.append(('Molecule' + '_' + 'f{self.mol_count}', smiles[i]))
            
            # Save and display the image
            img_nodes = drawer_nodes.GetDrawingText()
            img_edges = drawer_edges.GetDrawingText()
            img_nodes_edges = drawer_nodes_edges.GetDrawingText()
            img_mol = drawer_mol.GetDrawingText()
            
            img_nodes_path = "masking_test/node_importance/Molecule_" + f"{self.mol_count}" + ".png"
            img_edges_path = "masking_test/edge_importance/Molecule_" + f"{self.mol_count}" + ".png"
            img_nodes_edges_path = "masking_test/node_edge_importance/Molecule_" + f"{self.mol_count}" + ".png"
            img_mol_path = "masking_test/mols/Molecule_" + f"{self.mol_count}" + ".png"
            img_ann_nodes_path = "masking_test/annotated_nodes/Molecule_" + f"{self.mol_count}" + ".png"
            img_ann_edges_path = "masking_test/annotated_edges/Molecule_" + f"{self.mol_count}" + ".png"
            
            img_ann_nodes.save(img_ann_nodes_path)
            img_ann_edges.save(img_ann_edges_path)

            with open(img_nodes_path, "wb") as f:
                f.write(img_nodes)
                
            with open(img_edges_path, "wb") as f:
                 f.write(img_edges)
           
            with open(img_nodes_edges_path, "wb") as f:
                 f.write(img_nodes_edges)
                 
            with open(img_mol_path, "wb") as f:
                f.write(img_mol)
                        
            file_path = "masking_test/node_importance/molecule_SMILES.txt"
            # Writing list elements to a text file
            with open(file_path, 'a') as file:
                file.write("Molecule_" +  f"{self.mol_count}:" + f"{smiles[i]}" + "; Pred:" + f"{preds[i]}" +"; Label:" + f"{labels[i]}\n")
                
            file_path = "masking_test/mols/molecule_SMILES.txt"
            # Writing list elements to a text file
            with open(file_path, 'a') as file:
                file.write("Molecule_" +  f"{self.mol_count}:" + f"{smiles[i]}" + "; Pred:" + f"{preds[i]}" +"; Label:" + f"{labels[i]}\n")
            
            file_path = "masking_test/edge_importance/molecule_SMILES.txt"
            # Writing list elements to a text file
            with open(file_path, 'a') as file:
                file.write("Molecule_" +  f"{self.mol_count}:" + f"{smiles[i]}" + "; Pred:" + f"{preds[i]}" +"; Label:" + f"{labels[i]}\n")
                  
            file_path = "masking_test/node_edge_importance/molecule_SMILES.txt"
            # Writing list elements to a text file
            with open(file_path, 'a') as file:
                file.write("Molecule_" +  f"{self.mol_count}:" + f"{smiles[i]}" + "; Pred:" + f"{preds[i]}" +"; Label:" + f"{labels[i]}\n")
                          
        return []
        
    def mask_node_features(self, batch, node_idx):
       masked_batch = batch.clone()
       masked_batch.x[node_idx, :] = 0
       
       return masked_batch

    def mask_edge_features(self, batch, edge_idx):
        masked_batch = batch.clone()
        masked_batch.edge_attr[edge_idx, :] = 0
        
        return masked_batch

    def meta_evaluate(self, exp):
           
        graph_params = parameters_to_vector(self.gnn.parameters())
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        if self.baseline == 0:
            tr_params = parameters_to_vector(self.transformer.parameters())
        
        for test_task in range(self.test_tasks):
            
            support_set, query_set, smiles_ss, smiles_qs = sample_test(self.tasks, test_task, self.data, self.batch_size, self.n_support, self.n_query, exp)
            self.gnn.eval()
            if self.baseline == 0:
                self.transformer.eval()
                
            for k in range(0, self.k_test):
                
                graph_loss = torch.tensor([0.0]).to(device)
                
                if self.baseline == 0:
                    loss_logits = torch.tensor([0.0]).to(device)
                
                for batch_idx, (batch, batch2) in enumerate(zip(tqdm(support_set, desc="Iteration"), tqdm(smiles_ss, desc="Iteration"))):
                    
                    batch = batch.to(device)
                    graph_pred, emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    y = batch.y.view(graph_pred.shape)
                    loss_graph = self.loss(graph_pred.double(), y.to(torch.float64))
                    graph_loss += torch.sum(loss_graph)/graph_pred.size(dim=0)
                    
                    if self.baseline == 0:
                        
                        val_logit, emb = self.transformer(self.gnn.pool(emb, batch.batch))
                        loss_tr = self.loss_transformer(F.sigmoid(val_logit).double(), y.to(torch.float64))
                        loss_logits += torch.sum(loss_tr)/val_logit.size(dim=0)
                                             
                updated_grad, updated_params = self.update_graph_params(graph_loss, lr_update = self.lr_update)
                vector_to_parameters(updated_params, self.gnn.parameters())
                
                if self.baseline == 0:
                    updated_grad_tr, updated_tr_params = self.update_tr_params(loss_logits, lr_update = self.lr_update)
                    vector_to_parameters(updated_tr_params, self.transformer.parameters())
                   
            y_pred = []
            y_label = []
            self.mol_count = 0
            self.mol_D = []
                                  
            for batch_idx, (batch, batch2) in enumerate(zip(tqdm(query_set, desc="Iteration"), tqdm(smiles_qs, desc="Iteration"))):
                 
                 batch = batch.to(device)
                 
                 with torch.no_grad(): 
                     original_preds, emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                 if self.baseline == 0:
                     original_preds, emb = self.transformer(self.gnn.pool(emb, batch.batch))
                     
                 preds_or = parse_pred(original_preds).tolist()
                 preds_or = [item for sublist in preds_or for item in sublist]

                 labels = [0.0,1.0,1.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0] # laboratory results
 
                 y_pred.append(preds_or) 
                 y_label.append(labels)
                                                      
                 # mask node features
                 node_importances = torch.zeros(batch.x.size(0))
                 idx_importances = batch.batch.cpu().detach().numpy().tolist()
                 
                 for node_idx in range(batch.x.size(0)):
                                        
                      masked_batch = self.mask_node_features(batch, node_idx)
                      with torch.no_grad(): 
                          preds, emb = self.gnn(masked_batch.x, masked_batch.edge_index, masked_batch.edge_attr, masked_batch.batch)
                      if self.baseline == 0:
                          masked_preds, emb = self.transformer(self.gnn.pool(emb, batch.batch))
                          
                      masked_preds = F.sigmoid(masked_preds).tolist()
                      masked_preds = [item[0] for item in masked_preds]
                      
                      change_in_preds = [abs(a - b) for a, b in zip(labels, masked_preds)]
                                       
                      node_importances[node_idx] = change_in_preds[batch.batch[node_idx].item()] 

                 # Edge importance calculation
                 edge_importances = torch.zeros(batch.edge_index.size(1))
                 
                 for edge_idx in range(batch.edge_index.size(1)):
                     
                     masked_batch = self.mask_edge_features(batch, edge_idx)
                     with torch.no_grad():
                         masked_preds, emb = self.gnn(masked_batch.x, masked_batch.edge_index, masked_batch.edge_attr, masked_batch.batch)
                     if self.baseline == 0:
                         masked_preds, emb = self.transformer(self.gnn.pool(emb, batch.batch))
                     
                     masked_preds = F.sigmoid(masked_preds).tolist()
                     masked_preds = [item[0] for item in masked_preds]
                     
                     change_in_preds = [abs(a - b) for a, b in zip(labels, masked_preds)]
                     
                     edge_importances[edge_idx] = change_in_preds[batch.batch.tolist()[batch.edge_index.tolist()[0][edge_idx]]]
                
                 edge_idx_importances = []
               
                 for edge_idx in range(len(batch.edge_index.tolist()[0])): 
                     edge_idx_importances.append(batch.batch.tolist()[batch.edge_index.tolist()[0][edge_idx]])
                                      
                 node_importances = node_importances.tolist()
                 edge_importances = edge_importances.tolist()
                 
                 idx_importances = batch.batch.cpu().detach().numpy().tolist()
                 smiles = batch2  # The SMILES strings of the molecules in the batch 
                 
                 print("\nCompound SMILES: Prediction")
                 
                 pred_string = ['Mutagenic' if x[0] == 1 else 'Non-Mutagenic' for x in parse_pred(original_preds).tolist()]

                 output = {"\n" + batch2[i]: pred_string[i] + "\nProb. Mutagenic:" + str(F.sigmoid(original_preds).tolist()[i])[1:-1] + "\n" for i in range(len(batch2))}
                 
                 for key, value in output.items():
                     print(f"{key}: {value}")
                             
                 self.visualize_molecule(smiles, node_importances, edge_importances, idx_importances, edge_idx_importances, preds_or, labels)
                 
            vector_to_parameters(graph_params, self.gnn.parameters())
            if self.baseline == 0:
                vector_to_parameters(tr_params, self.transformer.parameters())

        return self.gnn.state_dict(), self.transformer.state_dict(), self.opt.state_dict(), self.meta_opt.state_dict()