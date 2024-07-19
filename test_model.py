import torch
from gnntr_test import GNNTR_eval

def save_ckp(state, is_best, checkpoint_dir, filename):
    
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)

dataset = "muta"
gnn = "gin" #gin, graphsage, gcn
support_set = 5
pretrained = "pre-trained/supervised_contextpred.pth" #supervised_contextpred.pth, graphsage_supervised_contextpred.pth, gcn_supervised_contextpred.pth
baseline = 0
device = "cuda:0"

# Meta-GTMP - Two module GNN-Transformer architecture for Ames mutagenicity prediction
# GraphSAGE
# GIN - Graph Isomorphism Network
# GCN - Standard Graph Convolutional Network

device = "cuda:0"      
model_eval = GNNTR_eval(dataset, gnn, support_set, pretrained, baseline)
model_eval.to(device)

print("Dataset -> Experimental Validation Candidate Compounds")

n = 1 # first experiment
N = 1 # number of experiments

for exp in range(1, N+1):
    
    gnn_model, transformer_model, gnn_opt, t_opt = model_eval.meta_evaluate(exp)


   

    