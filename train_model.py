import torch
import numpy as np
from gnntr_train import GNNTR

def save_ckp(state, is_best, checkpoint_dir, filename):
    if is_best == True:
        f_path = checkpoint_dir + filename
        torch.save(state, f_path)

dataset = "muta"
gnn = "gin" #gin, graphsage, gcn
support_set = 5
pretrained = "pre-trained/supervised_contextpred.pth" #supervised_contextpred.pth, graphsage_supervised_contextpred.pth, gcn_supervised_contextpred.pth
baseline = 0
device = "cuda:0"

torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed_all(1)

model = GNNTR(dataset, gnn, support_set, pretrained, baseline)
model.to(device)

if dataset == "muta":
    
    roc = 0
    f1s = 0
    prs = 0
    sns = 0
    sps = 0
    acc = 0
    bacc = 0
    
    best_auroc = 0

    labels =  ['Overall']
    
for epoch in range(1, 2001):
    
    is_best = False
    
    model.meta_train()
    
    [roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores], gnn_model, transformer_model, gnn_opt, t_opt = model.meta_test() 
    
    print(roc_scores, f1_scores, p_scores, sn_scores, sp_scores, acc_scores, bacc_scores)

    checkpoint_gnn = {
            'epoch': epoch + 1,
            'state_dict': gnn_model,
            'optimizer': gnn_opt
    }
    
    checkpoint_transformer = {
            'epoch': epoch + 1,
            'state_dict': transformer_model,
            'optimizer': t_opt
    }

    checkpoint_dir = 'checkpoints/checkpoints-GT/'
    
    roc_scores = roc_scores[0]
    f1_scores = f1_scores[0]
    p_scores = p_scores[0]
    sn_scores = sn_scores[0]
    sp_scores = sp_scores[0]
    acc_scores = acc_scores[0]
    bacc_scores = bacc_scores[0]
         
    if roc < roc_scores:
        roc = roc_scores
        is_best = True
        print(roc)

    if f1s < f1_scores:
        f1s = f1_scores

    if prs < p_scores:
        prs = p_scores

    if sns < sn_scores:
        sns = sn_scores
    
    if sps < sp_scores:
        sps = sp_scores

    if acc < acc_scores:
        acc = acc_scores
    
    if bacc < bacc_scores:
        bacc = bacc_scores

    filename = "results-exp/Meta-GTMP-5.txt"
    file = open(filename, "a")
    file.write("[ROC-AUC, F1-score, Precision, Sensitivity, Specificity, Accuracy, Balanced Accuracy]:\t")
    #file.write(str(exp))
    file.write("[" + str(roc_scores) + "," + str(f1_scores) + "," + str(p_scores) + "," + str(sn_scores) + "," + str(sp_scores) + "," + str(acc_scores) + "," + str(bacc_scores) + "]" + "\t ; ")
    file.write("[" + str(roc) + "," +  str(f1s) + "," + str(prs) + "," + str(sns) + "," + str(sps) + "," + str(acc) + "," + str(bacc) + "]") 
    file.write("\n")
    file.close()
    
    if baseline == 0:
        save_ckp(checkpoint_gnn, is_best, checkpoint_dir, "/Meta-GTMP_GNN_muta_5.pt")
        save_ckp(checkpoint_transformer, is_best, checkpoint_dir, "/Meta-GTMP_Transformer_muta_5.pt")

    if baseline == 1:
        save_ckp(checkpoint_gnn, is_best, checkpoint_dir, "/GIN_GNN_muta_5.pt")
        
   