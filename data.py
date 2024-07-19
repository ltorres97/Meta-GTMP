import os
import sys
import torch
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat
import random

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


def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

def datasets(data):
    if data == "muta":
        return[[2782, 1676], [2721, 2096], [587, 226], [2103, 436], [1779, 365], [231, 3103]]

def random_sampler(D, d, t, k, n, train, seed=None):
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
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

    return S, Q


def random_sampler_mask(D, d, t, k, n, train, seed=None):
        
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    data = datasets(d)
    
    s_pos = random.sample(range(0,data[t][0]), k)
    s_neg = random.sample(range(data[t][0],len(D)), k)
    s = s_pos + s_neg
    random.shuffle(s)
    
    samples = [i for i in range(0, len(D)) if i not in s]
    
    if train == True:     
        q = random.sample(samples, n)
    else:
        #random.shuffle(samples)
        q = samples
 
    S = D[torch.tensor(s)]
    Q = D[torch.tensor(q)]
    
    smiles_s = [D.smiles_list[i] for i in s]
    smiles_q = [D.smiles_list[i] for i in q]
    
    print(len(S))
    print(len(smiles_s))
    
    return S, Q, smiles_s, smiles_q


def random_sampler_test(D, D_t, d, t, k, n, train, seed=None):
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
    data = datasets(d)
    
    s_pos = random.sample(range(0,data[t][0]), k)
    s_neg = random.sample(range(data[t][0],len(D)), k)
    s = s_pos + s_neg
    random.shuffle(s)
    
    samples = [i for i in range(0, len(D_t))]
    
    if train == True:     
        q = random.sample(samples, n)
    else:
        #random.shuffle(samples)
        q = samples
 
    S = D[torch.tensor(s)]
    Q = D_t[torch.tensor(q)]
    
    smiles_s = [D.smiles_list[i] for i in s]
    smiles_q = [D_t.smiles_list[i] for i in q]
        
    return S, Q, smiles_s, smiles_q

        
def split_into_directories(data):

    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root_dir)
    file = open(os.path.join(root_dir, 'Data/{}/original/{}.csv'.format(data,data)), 'r').readlines()[1:]
    np.random.shuffle(file)
    
    T = {}
    C = {}
  
    s = 0

    if data == "val":
        
        T = []
        C = {}
        
        for k,j  in enumerate(file):
            sample = j.split(";")
           
            T.append(sample[1].rstrip()[:-1])
            
        print(T)
        d = 'Data/' + data + "/pre-processed/" + "task_1"
        os.makedirs(d, exist_ok=True)
        os.makedirs(d + "/raw", exist_ok=True)
        os.makedirs(d + "/processed", exist_ok=True)
        
        with open(d + "/raw/" + data + "_task_1", "wb") as fp:   #Pickling
            pickle.dump(T, fp)
    
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
        
        
        with open("sa_overall.smi", 'w') as f:
            f.write("\n".join(str(i) for i in mcss_SMILES))
        
        with open("sa_overall.bio", 'w') as f:
            f.write("\n".join(str(i) for i in mcss_label))
                
        print(s)
        print(C)
        print(T)
                
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
                 dataset='muta',
                 empty=False):
        
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
        
        if self.dataset == "val":
              smiles_list, rdkit_mol_objs, labels = \
                  _load_val_dataset(self.raw_paths[0])
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

def _load_val_dataset(input_path):
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
        smiles_list.append(l)
    
    print(smiles_list)
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
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
    if data == "val":
         T = 1

    for task in range(T):
        build = MoleculeDataset(root + "task_" + str(task+1), dataset=data)

if __name__ == "__main__":

    split_into_directories("val")
    dataset("val")
    #structure_alerts()