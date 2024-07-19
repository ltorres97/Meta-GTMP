import os
import sys
import numpy as np
import random
from rdkit import Chem
from sklearn import model_selection, svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()   

def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
    
        missingVal is used if the descriptor cannot be calculated
    '''
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

def baselines(data):

    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root_dir)
    file = open(os.path.join(root_dir, 'Data/{}/original/{}.csv'.format(data,data)), 'r').readlines()[1:]
    np.random.shuffle(file)
    
    T = {}
    C = {}
  
    s = 0
           
    if data == "muta":
        
        mcss_SMILES = []
        mcss_label = []
        for k,j in enumerate(file):
          sample = j.split(",")
          count=0
          smile=""
          sample_processed = []
          
          for i in range(0,len(sample)):
            if check_int(sample[i]) == False and (sample[i].replace('.','',1).isdigit() or sample[i][1:].replace('.','',1).isdigit()) and count == 0:
              sample_processed = sample[i:]
              smile = sample[i-1]
              count = 1
          
          if sample[0] == '5139':
            smile = sample[3]
            sample_processed = sample[4:]
          
          if sample[0] == '5005':
            smile = sample[5]
            sample_processed = sample[6:]
          
          if sample[0] == '2164':
            smile = sample[7]
            sample_processed = sample[8:]

          if sample[0] == '2577':
            smile = sample[5]
            sample_processed = sample[6:]
          
          if sample[0] == '5140':
            smile = sample[3]
            sample_processed = sample[4:]
          
          if sample[0]  == '2510':
            smile = sample[5]
            sample_processed = sample[6:]

          if sample[0] == '2990':
            smile = sample[5]
            sample_processed = sample[6:]
          
          sample_final = sample_processed[-7:]
          sample_final = sample_final[:-1]
          #sample_final = [smile] + sample_final
          #print(sample_final)

          m = Chem.MolFromSmiles(smile)
          if m is None:
            print('invalid')
            print(sample)

          for i in range(6):
                if i not in T:
                    T[i] = [[],[]]
                if i not in C:
                    C[i] = [0,0]
                if sample_final[i] == '0':
                    T[i][0].append(smile)
                    C[i][0]+=1
                elif sample_final[i] == '1':
                    T[i][1].append(smile)
                    C[i][1]+=1
                if sample_final[i] == '-1' and i == 5:
                    s+=1
                if (sample_final[i] == '0' or sample_final[i] == '1') and i == 5:
                    mcss_SMILES.append(smile)   
                    mcss_label.append(sample_final[i])
        
        #print(s)
        #print(T[5])
        #print(len(T[5]))
        #print(C)
               
        smiles_data_neg =  T[5][0]
        smiles_data_pos =  T[5][1]
        
        smiles_neg = []
        smiles_neg_label = []
        smiles_pos=[]
        smiles_pos_label = []
        
        #Features - rdkit.Chem.rdchem
        """
        rdkit.Chem.rdchem.Atom - The class to store Atoms. Note that, though it is possible to create one, having an Atom on its own (i.e not associated with a molecule) is not particularly useful.
        rdkit.Chem.rdchem.AtomMonomerInfo - The class to store monomer information attached to Atoms
        rdkit.Chem.rdchem.AtomMonomerType
        rdkit.Chem.rdchem.AtomPDBResidueInfo - The class to store PDB residue information attached to Atoms
        rdkit.Chem.rdchem.Bond - The class to store Bonds. Note: unlike Atoms, is it currently impossible to construct Bonds from Python.
        rdkit.Chem.rdchem.BondDir - The class to store bond direction information attached to Bonds
        rdkit.Chem.rdchem.BondStereo
        rdkit.Chem.rdchem.BondType - The class to store bond type information attached to Bonds
        rdkit.Chem.rdchem.ChiralType
        rdkit.Chem.rdchem.CompositeQueryType
        rdkit.Chem.rdchem.Conformer - The class to store 2D or 3D conformation of a molecule
        rdkit.Chem.rdchem.EditableMol - an editable molecule class
        rdkit.Chem.rdchem.FixedMolSizeMolBundle - A class for storing groups of related molecules. Here related means that the molecules have to have the same number of atoms.
        rdkit.Chem.rdchem.HybridizationType
        rdkit.Chem.rdchem.Mol - The Molecule class
        rdkit.Chem.rdchem.MolBundle - A class for storing groups of related molecules
        rdkit.Chem.rdchem.PeriodicTable - A class which stores information from the Periodic Table
        rdkit.Chem.rdchem.PropertyPickleOptions
        rdkit.Chem.rdchem.QueryAtom - The class to store QueryAtoms
        rdkit.Chem.rdchem.QueryBond - The class to store QueryBonds
        rdkit.Chem.rdchem.RWMol - The RW molecule class (read/write). This class is a more-performant version of the EditableMolecule class in that it is a ‘live’ molecule and shares the interface from the Mol class. All changes are performed without the need to create a copy of the molecule using GetMol() (this is still available, however).
        rdkit.Chem.rdchem.ResonanceFlags
        rdkit.Chem.rdchem.ResonanceMolSupplier - A class which supplies resonance structures (as mols) from a mol.
        rdkit.Chem.rdchem.ResonanceMolSupplierCallback 
        rdkit.Chem.rdchem.RingInfo - contains information about a molecule’s rings
        rdkit.Chem.rdchem.StereoDescriptor
        rdkit.Chem.rdchem.StereoGroup - A collection of atoms with a defined stereochemical relationship. Used to help represent a sample with unknown stereochemistry, or that is a mix of diastereomers.
        rdkit.Chem.rdchem.StereoGroupType
        rdkit.Chem.rdchem.StereoGroup_vect
        rdkit.Chem.rdchem.StereoInfo
        rdkit.Chem.rdchem.StereoSpecified
        rdkit.Chem.rdchem.StereoType
        rdkit.Chem.rdchem.SubstanceGroup - A collection of atoms and bonds with associated properties
        rdkit.Chem.rdchem.SubstanceGroupAttach - AttachPoint for a SubstanceGroup
        rdkit.Chem.rdchem.SubstanceGroupCState
        rdkit.Chem.rdchem.SubstanceGroup_VECT
        rdkit.Chem.rdchem.SubstructMatchParameters     
        """
        
        descriptors_list = [x[0] for x in Descriptors._descList]
        #print(descriptors_list)
        
        for i in smiles_data_neg:
            
            #Molecular Graph
            mol = Chem.MolFromSmiles(i)
            #smiles_neg.append(np.array(RDKFingerprint(mol))) #RDKit Fingerprints
            smiles_neg.append((np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2)))) #Morgan Fingerprints
            smiles_neg_label.append(0)
            
        for i in smiles_data_pos:
            
            #Molecular Graph
            mol = Chem.MolFromSmiles(i)
            #smiles_pos.append(np.array(RDKFingerprint(mol))) #RDKit Fingerprints
            smiles_pos.append((np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2)))) #Morgan Fingerprints
            smiles_pos_label.append(1)
                
        all_smiles = smiles_neg + smiles_pos
        all_labels = smiles_neg_label + smiles_pos_label
        
        #print(all_smiles)
        #print(all_labels)
        
        roc_SVM = []
        acc_SVM = []
        f1s_SVM = []
        prs_SVM = []
        sns_SVM = []
        sps_SVM = []
        
        roc_RF = []
        acc_RF = []
        f1s_RF = []
        prs_RF = []
        sns_RF = []
        sps_RF = []
        
        roc_KNN = []
        acc_KNN = []
        f1s_KNN = []
        prs_KNN = []
        sns_KNN = []
        sps_KNN = []
        
        roc_GP = []
        acc_GP = []
        f1s_GP = []
        prs_GP = []
        sns_GP = []
        sps_GP = []
        
        for i in range(2, 32):
            
            seed = i
            random.seed(seed)
           
            Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(all_smiles,all_labels, test_size=0.3, shuffle=True, random_state=seed)
            
            #################### SVM ####################
                        
            SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', class_weight= 'balanced', random_state=seed)
            
            SVM.fit(Train_X,Train_Y)
            
            predictions_SVM = SVM.predict(Test_X)
            
            roc_SVM.append(roc_auc_score(Test_Y, predictions_SVM))
            acc_SVM.append(accuracy_score(Test_Y, predictions_SVM))
            f1s_SVM.append(f1_score(Test_Y, predictions_SVM))
            prs_SVM.append(precision_score(Test_Y, predictions_SVM))
            sns_SVM.append(recall_score(Test_Y, predictions_SVM))
            tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_SVM).ravel()
            sp = tn/(tn+fp)
            sps_SVM.append(sp)
            
            #################### RF ####################
            
            RF = RandomForestClassifier(max_depth=10, n_estimators=100, class_weight='balanced', random_state=seed)
            
            RF.fit(Train_X,Train_Y)
            
            predictions_RF = RF.predict(Test_X)
            
            roc_RF.append(roc_auc_score(Test_Y, predictions_RF))
            acc_RF.append(accuracy_score(Test_Y, predictions_RF))
            f1s_RF.append(f1_score(Test_Y, predictions_RF))
            prs_RF.append(precision_score(Test_Y, predictions_RF))
            sns_RF.append(recall_score(Test_Y, predictions_RF))
            tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_RF).ravel()
            sp = tn/(tn+fp)
            sps_RF.append(sp)
            
            #################### GP ####################
            
            GP = GaussianProcessClassifier(kernel = 1.0 * RBF(1.0), max_iter_predict = 100, random_state=seed)
            
            GP.fit(Train_X,Train_Y)
            
            predictions_GP = GP.predict(Test_X)
            
            roc_GP.append(roc_auc_score(Test_Y, predictions_GP))
            acc_GP.append(accuracy_score(Test_Y, predictions_GP))
            f1s_GP.append(f1_score(Test_Y, predictions_GP))
            prs_GP.append(precision_score(Test_Y, predictions_GP))
            sns_GP.append(recall_score(Test_Y, predictions_GP))
            tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_GP).ravel()
            sp = tn/(tn+fp)
            sps_GP.append(sp)
                        
            #################### KNN ####################
            
            KNN = KNeighborsClassifier(n_neighbors=3)
            
            KNN.fit(Train_X,Train_Y)
            
            predictions_KNN = KNN.predict(Test_X)
            
            roc_KNN.append(roc_auc_score(Test_Y, predictions_KNN))
            acc_KNN.append(accuracy_score(Test_Y, predictions_KNN))
            f1s_KNN.append(f1_score(Test_Y, predictions_KNN))
            prs_KNN.append(precision_score(Test_Y, predictions_KNN))
            sns_KNN.append(recall_score(Test_Y, predictions_KNN))
            tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_KNN).ravel()
            sp = tn/(tn+fp)
            sps_KNN.append(sp)
            
        filename = "results-exp/ml-baselines_muta_morgan.txt"
        file = open(filename, "a")
                                   
        file.write("\nMean SVM Accuracy Score -> " + str(np.mean(acc_SVM)) + "+-"+ str(np.std(acc_SVM)))
        file.write("\nMean SVM ROC-AUC Score -> "+ str(np.mean(roc_SVM)) + "+-"+ str(np.std(roc_SVM)))
        file.write("\nMean SVM F1-Score -> "+ str(np.mean(f1s_SVM))+ "+-" + str(np.std(f1s_SVM)))
        file.write("\nMean SVM Precision Score -> "+ str(np.mean(prs_SVM))+ "+-"+ str(np.std(prs_SVM)))
        file.write("\nMean SVM Sensitivity Score -> "+ str(np.mean(sns_SVM))+ "+-"+ str(np.std(sns_SVM)))
        file.write("\nMean SVM Specificity Score -> "+ str(np.mean(sps_SVM))+ "+-"+ str(np.std(sps_SVM)))
        file.write("\n")
        file.write("\nMean RF Accuracy Score -> "+ str(np.mean(acc_RF))+ "+-"+ str(np.std(acc_RF)))
        file.write("\nMean RF ROC-AUC Score -> "+ str(np.mean(roc_RF))+ "+-"+ str(np.std(roc_RF)))
        file.write("\nMean RF F1-Score -> "+ str(np.mean(f1s_RF))+ "+-"+ str(np.std(f1s_RF)))
        file.write("\nMean RF Precision Score -> "+ str(np.mean(prs_RF))+ "+-"+ str(np.std(prs_RF)))
        file.write("\nMean RF Sensitivity Score -> "+ str(np.mean(sns_RF))+ "+-"+ str(np.std(sns_RF)))
        file.write("\nMean RF Specificity Score -> "+ str(np.mean(sps_RF))+ "+-"+ str(np.std(sps_RF)))
        file.write("\n")
        file.write("\nMean GP Accuracy Score -> "+ str(np.mean(acc_GP))+ "+-"+ str(np.std(acc_GP)))
        file.write("\nMean GP ROC-AUC Score -> "+ str(np.mean(roc_GP))+ "+-"+ str(np.std(roc_GP)))
        file.write("\nMean GP F1-Score -> "+ str(np.mean(f1s_GP))+ "+-"+ str(np.std(f1s_GP)))
        file.write("\nMean GP Precision Score -> "+ str(np.mean(prs_GP))+ "+-"+ str(np.std(prs_GP)))
        file.write("\nMean GP Sensitivity Score -> "+ str(np.mean(sns_GP))+ "+-"+ str(np.std(sns_GP)))
        file.write("\nMean GP Specificity Score -> "+ str(np.mean(sps_GP))+ "+-"+ str(np.std(sps_GP)))
        file.write("\n")
        file.write("\nMean KNN Accuracy Score -> "+ str(np.mean(acc_KNN))+ "+-"+ str(np.std(acc_KNN)))
        file.write("\nMean KNN ROC-AUC Score -> "+ str(np.mean(roc_KNN))+ "+-"+ str(np.std(roc_KNN)))
        file.write("\nMean KNN F1-Score -> "+ str(np.mean(f1s_KNN))+ "+-"+ str(np.std(f1s_KNN)))
        file.write("\nMean KNN Precision Score -> "+ str(np.mean(prs_KNN))+ "+-"+ str(np.std(prs_KNN)))
        file.write("\nMean KNN Sensitivity Score -> "+ str(np.mean(sns_KNN))+ "+-"+ str(np.std(sns_KNN)))
        file.write("\nMean KNN Specificity Score -> "+ str(np.mean(sps_KNN))+ "+-"+ str(np.std(sps_KNN)))
        
        
if __name__ == "__main__":
    # compute baseline results - SVM, RF, GP, KNN
    baselines("muta")