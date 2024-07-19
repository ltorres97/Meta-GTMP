import torch
from gnntr_mask import GNNTR_eval
import statistics

def save_ckp(state, is_best, checkpoint_dir, filename):
    
    f_path = checkpoint_dir + filename
    torch.save(state, f_path)

def save_result(epoch, N, exp, filename):
            
    file = open(filename, "a")
    
    if epoch < N:
        file.write("Results: " + "\t")
        file.write(str(exp) + "\t")
    if epoch == N:
        file.write("Results: " + "\t")
        file.write(str(exp) + "\n")
        file.write(str(epoch) + " Support Sets: (Mean:"+ str(statistics.mean(exp)) +", SD:" +str(statistics.stdev(exp)) + ") | \t")
    file.write("\n")
    file.close()
    
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
 
model_eval = GNNTR_eval(dataset, gnn, support_set, pretrained, baseline)
model_eval.to(device)

print("Dataset:", dataset)

roc_auc_list = []

if dataset == "muta":
    roc = []
    f1s = []
    prs = []
    sns = []
    sps = []
    acc = []
    bacc = []
    labels =  ['Overall Ames Mutagenicity']
 
n = 1 # first experiment
N = 1 # number of experiments

for exp in range(n, N+1):
    
    gnn_model, transformer_model, gnn_opt, t_opt = model_eval.meta_evaluate(exp)
