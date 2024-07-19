import torch
from gnntr_eval import GNNTR_eval 
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
N = 30 # number of experiments

# For evaluation experiments: n = 1; N = 30 (random seeds = 2,...,31) -> gnntr_eval.py (seed = exp + 1)
# For t-sne embedding visualization: n = N = 39 (random seed = 40) -> gnntr_eval.py (seed = exp + 1)
# For substructure masking calculation: n = N = 1 (random seed = 2) -> emb_eval.py (seed = exp + 1)

for exp in range(n, N+1):
    
    [roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores], gnn_model, transformer_model, gnn_opt, t_opt = model_eval.meta_evaluate(exp)

    if exp <= N:
      roc.append(round(roc_scores,4))
      f1s.append(round(f1_scores,4))
      prs.append(round(p_scores,4))
      sns.append(round(sn_scores,4))
      sps.append(round(sp_scores,4))
      acc.append(round(acc_scores,4))
      bacc.append(round(bacc_scores,4))
    
    save_result(exp, N, roc, "results-exp/Meta-GTMP/roc-Meta-GTMP_muta_5.txt")
    save_result(exp, N, f1s, "results-exp/Meta-GTMP/f1s-Meta-GTMP_muta_5.txt")
    save_result(exp, N, prs, "results-exp/Meta-GTMP/prs-Meta-GTMP_muta_5.txt")
    save_result(exp, N, sns, "results-exp/Meta-GTMP/sns-Meta-GTMP_muta_5.txt")
    save_result(exp, N, sps, "results-exp/Meta-GTMP/sps-Meta-GTMP_muta_5.txt")
    save_result(exp, N, acc, "results-exp/Meta-GTMP/acc-Meta-GTMP_muta_5.txt")
    save_result(exp, N, bacc, "results-exp/Meta-GTMP/bacc-Meta-GTMP_muta_5.txt")
    
    #save_result(exp, N, roc, "results-exp/ST-models/ST1/roc-ST1-Meta-GTMP_muta_5.txt")
    #save_result(exp, N, f1s, "results-exp/ST-models/ST1/f1s-ST1-Meta-GTMP_muta_5.txt")
    #save_result(exp, N, prs, "results-exp/ST-models/ST1/prs-ST1-Meta-GTMP_muta_5.txt")
    #save_result(exp, N, sns, "results-exp/ST-models/ST1/sns-ST1-Meta-GTMP_muta_5.txt")
    #save_result(exp, N, sps, "results-exp/ST-models/ST1/sps-ST1-Meta-GTMP_muta_5.txt")
    #save_result(exp, N, acc, "results-exp/ST-models/ST1/acc-ST1-Meta-GTMP_muta_5.txt")
    #save_result(exp, N, bacc, "results-exp/ST-models/ST1/bacc-ST1-Meta-GTMP_muta_5.txt")
    
    