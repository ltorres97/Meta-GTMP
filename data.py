"""
---------------------------------------------------------------------------------------
Code Implementation to pre-process the data and perform the transformation of SMILES
strings into molecular graphs using RDKit.Chem.
Run this script to pre-process data saved in the Data/[DatasetName]/pre-processed
/[Task]/processed file.
This code is based on Strategies for Pre-training Graph Neural Networks of Hu et al.
Paper: Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., Leskovec, J.:
Strategies for Pre-training Graph Neural Networks. arXiv (2019). 
https://arxiv.org/abs/1905.12265
---------------------------------------------------------------------------------------
"""

import os
import sys
import torch
import pickle
import collections
import math
import ast
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
import torch
import random
import bioalerts
from bioalerts import LoadMolecules, Alerts, FPCalculator

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def SmilesMCStoGridImage(smiles, align_substructure = False, verbose = True, **kwargs):
     """
     Convert a list (or dictionary) of SMILES strings to an RDKit grid image of the maximum common substructure (MCS) match between them

     :returns: RDKit grid image, and (if verbose=True) MCS SMARTS string and molecule, and list of molecules for input SMILES strings
     :rtype: RDKit grid image, and (if verbose=True) string, molecule, and list of molecules
     :param molecules: The SMARTS molecules to be compared and drawn
     :type molecules: List of (SMARTS) strings, or dictionary of (SMARTS) string: (legend) string pairs
     :param align_substructure: Whether to align the MCS substructures when plotting the molecules; default is True
     :type align_substructure: boolean
     :param verbose: Whether to return verbose output (MCS SMARTS string and molecule, and list of molecules for input SMILES strings); default is False so calling this function will present a grid image automatically
     :type verbose: boolean
     """
     mols = [Chem.MolFromSmiles(smile) for smile in smiles]
     res = rdFMCS.FindMCS(mols, **kwargs)
     mcs_smarts = res.smartsString
     mcs_mol = Chem.MolFromSmarts(res.smartsString)
     smarts = res.smartsString
     smart_mol = Chem.MolFromSmarts(smarts)
     smarts_and_mols = [smart_mol] + mols

     smarts_legend = "Max. substructure match"

     # If user supplies a dictionary, use the values as legend entries for molecules
     if isinstance(smiles, dict):
          mol_legends = [smiles[molecule] for molecule in smiles]
     else:
          mol_legends = ["" for mol in mols]

     legends =  [smarts_legend] + mol_legends
    
     matches = [""] + [mol.GetSubstructMatch(mcs_mol) for mol in mols]

     subms = [x for x in smarts_and_mols if x.HasSubstructMatch(mcs_mol)]

     Chem.rdDepictor.Compute2DCoords(mcs_mol)

     if align_substructure:
          for m in subms:
               _ = Chem.rdDepictor.GenerateDepictionMatching2DStructure(m, mcs_mol)

     drawing = Draw.MolsToGridImage(smarts_and_mols, highlightAtomLists=matches, legends=legends)
     drawing.save('submol.png')
     
     if verbose:
          return drawing, mcs_smarts, mcs_mol, mols
     else:
          return drawing

def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

def datasets(data):
    if data == "muta":
        return[[2782, 1676], [2721, 2096], [587, 226], [2103, 436], [1779, 365], [231, 3103]]

def random_sampler(D, d, t, k, n, train):
    
    data = datasets(d)
    
    s_pos = random.sample(range(0,data[t][0]), k)
    s_neg = random.sample(range(data[t][0],len(D)), k)
    s = s_pos + s_neg
    random.shuffle(s)
    
    samples = [i for i in range(0, len(D)) if i not in s]
    
    if train == True:     
        q = random.sample(samples, n)
    else:
        random.shuffle(samples)
        q = samples
 
    S = D[torch.tensor(s)]
    Q = D[torch.tensor(q)]
    
    smiles_s = [D.smiles_list[i] for i in s]
    smiles_q = [D.smiles_list[i] for i in q]
    
    print(len(S))
    print(len(smiles_s))
    
    return S, Q, smiles_s, smiles_q


def split_into_directories(data):

    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root_dir)
    file = open(os.path.join(root_dir, 'Data/{}/original/{}.csv'.format(data,data)), 'r').readlines()[1:]
    np.random.shuffle(file)
    
    T = {}
    C = {}
  
    kk = 0
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
                    kk+=1
                if (sample_final[i] == '0' or sample_final[i] == '1') and i == 5:
                    mcss_SMILES.append(smile)   
                    mcss_label.append(sample_final[i])
        
        
        with open("sa_overall.smi", 'w') as f:
            f.write("\n".join(str(i) for i in mcss_SMILES))
        
        with open("sa_overall.bio", 'w') as f:
            f.write("\n".join(str(i) for i in mcss_label))
            
        #drawing, mcs_smarts, mcs_mol, mols = SmilesMCStoGridImage(mcss_SMILES[:20])
        molecules = bioalerts.LoadMolecules.LoadMolecules("sa_overall.smi",name_field=None)
        molecules.ReadMolecules()
        
        mol_bio = np.genfromtxt('sa_overall.bio')
        arr = np.arange(0,len(mol_bio))
        mask = np.ones(arr.shape,dtype=bool)
        mask[molecules.molserr]=0
        mol_bio = mol_bio[mask]
        print(len(mol_bio))
        print(len(molecules.mols))
        
        stride = int(len(molecules.mols) * 0.9)
        training = molecules.mols[0:stride]
        test = molecules.mols[stride:len(molecules.mols)]
        print(len(molecules.mols), len(test), len(training))
        
        bio_training = mol_bio[0:stride]
        bio_test = mol_bio[stride:len(molecules.mols)]
        print(len(mol_bio), len(bio_test), len(bio_training))
        
        training_dataset_info = bioalerts.LoadMolecules.GetDataSetInfo(name_field=None)
        training_dataset_info.extract_substructure_information(radii=[2,3,4],mols=training)
        
        Alerts_categorical = bioalerts.Alerts.CalculatePvaluesCategorical(max_radius=4)

        Alerts_categorical.calculate_p_values(mols=test,
                                              substructure_dictionary=training_dataset_info.substructure_dictionary,
                                              bioactivities=bio_training,
                                              mols_ids=training_dataset_info.mols_ids[0:stride],
                                              threshold_nb_substructures = 50,
                                              threshold_pvalue = 0.05,
                                              threshold_frequency = 0.7,
                                              Bonferroni=True)
        
        Alerts_categorical.XlSXOutputWriter(Alerts_categorical.output, 
                                            'sa_overall.xlsx')
        
        print(kk)
        #print(T)
        print(C)
        
        
        for t in T:
            d = 'Data/' + data + "/pre-processed/" + "task_" +str(t+1)
            os.makedirs(d, exist_ok=True)
            os.makedirs(d + "/raw", exist_ok=True)
            os.makedirs(d + "/processed", exist_ok=True)
            
            with open(d + "/raw/" + data + "_task_" + str(t+1), "wb") as fp:   #Pickling
                pickle.dump(T[t], fp)

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='tox21',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices, self.smiles_list = torch.load(self.processed_paths[0])


    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        
        return data


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []
        if self.dataset == 'muta':
             smiles_list, rdkit_mol_objs, labels = \
                 _load_muta_dataset(self.raw_paths[0])
             for i in range(len(smiles_list)):
                 #print(i)
                 rdkit_mol = rdkit_mol_objs[i]
                 #print(smiles_list[i])
                 data = mol_to_graph_data_obj_simple(rdkit_mol)
                 # manually add mol id
                 data.id = torch.tensor(
                     [i])  # id here is the index of the mol in
                 # the dataset
                 data.y = torch.tensor(labels[i, :])
                 data_list.append(data)
                 data_smiles_list.append(smiles_list[i])
            
        else:
            raise ValueError('Invalid dataset name')
        
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'processed.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices, data_smiles_list), self.processed_paths[0])


def merge_dataset_objs(dataset_1, dataset_2):
    """
    Naively merge 2 molecule dataset objects, and ignore identities of
    molecules. Assumes both datasets have multiple y labels, and will pad
    accordingly. ie if dataset_1 has obj_1 with y dim 1310 and dataset_2 has
    obj_2 with y dim 128, then the resulting obj_1 and obj_2 will have dim
    1438, where obj_1 have the last 128 cols with 0, and obj_2 have
    the first 1310 cols with 0.
    :return: pytorch geometric dataset obj, with the x, edge_attr, edge_index,
    new y attributes only
    """
    d_1_y_dim = dataset_1[0].y.size()[0]
    d_2_y_dim = dataset_2[0].y.size()[0]

    data_list = []
    # keep only x, edge_attr, edge_index, padded_y then append
    for d in dataset_1:
        old_y = d.y
        new_y = torch.cat([old_y, torch.zeros(d_2_y_dim, dtype=torch.long)])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    for d in dataset_2:
        old_y = d.y
        new_y = torch.cat([torch.zeros(d_1_y_dim, dtype=torch.long), old_y.long()])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    # create 'empty' dataset obj. Just randomly pick a dataset and root path
    # that has already been processed
    new_dataset = MoleculeDataset(root='dataset/chembl_with_labels',
                                  dataset='chembl_with_labels', empty=True)
    # collate manually
    new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)

    return new_dataset

def create_circular_fingerprint(mol, radius, size, chirality):
    """
    :param mol:
    :param radius:
    :param size:
    :param chirality:
    :return: np array of morgan fingerprint
    """
    fp = GetMorganFingerprintAsBitVect(mol, radius,
                                       nBits=size, useChirality=chirality)
    return np.array(fp)


def _load_muta_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    with open(input_path, 'rb') as f:
       binary_list = pickle.load(f)

    print(binary_list)
  
    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i[:-1])
    
    rdkit_mol_objs_list = []
    for s in smiles_list:
        rdkit_mol_objs_list.append(Chem.MolFromSmiles(s))
        
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 
    
    

    return smiles_list, rdkit_mol_objs_list, labels


def _load_muta_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    with open(input_path, 'rb') as f:
       binary_list = pickle.load(f)

    print(binary_list)
    smiles_list = []
    for l in binary_list:
        for i in l:
            smiles_list.append(i)
    
    print(smiles_list)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = np.zeros((len(smiles_list),1), dtype=int)
    labels[len(binary_list[0]):,0] = 1 

    return smiles_list, rdkit_mol_objs_list, labels

def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False

def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list

def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def dataset(data):
    root = "Data/" + data + "/pre-processed/"
    T = 6
    if data == "muta":
        T = 6

    for task in range(T):
        build = MoleculeDataset(root + "task_" + str(task+1), dataset=data)

if __name__ == "__main__":
    # split data in mutiple data directories and convert to RDKit.Chem graph structures
    split_into_directories("muta")
    dataset("muta")