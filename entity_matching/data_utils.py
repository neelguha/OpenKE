# Define dataset and data loader for entity matching utility functions 

import os, json
import numpy as np 

from torch.utils import data
import torch

def load_openKE_dict(fdir, source):
    ''' Loads dict mapping entity IDs to indices. 
    '''
    fpath = os.path.join(fdir, 'OpenKE_%s' % source, 'entity2id.txt')
    eid2id = {}
    with open(fpath, 'r') as in_file:
        for line in in_file:
            items = line.strip().split("\t")
            if len(items) == 2:
                eid2id[items[0]] = int(items[1])
    return eid2id

def convert_matches(matches, mbz2index, wd2index):
    ''' Converts pairs of MBZ-WD ids to corresponding indices in embedding 
    matrices
    ''' 
    indexed_matches = []
    for mbz, wd in matches:
        if wd == "None":
            t = (mbz2index[mbz], -1)
        else:
            t = (mbz2index[mbz], wd2index[wd])
        indexed_matches.append(t)
    return indexed_matches

def load_matches(fdir, match_type):
    ''' Loads pairs of matching MBZ-WD IDs. 
    '''
    fpath = os.path.join(fdir, '%s_matches.txt' % match_type)
    matches = []
    with open(fpath, 'r') as in_file:
        for line in in_file:
            items = line.strip().split("\t")
            if len(items) == 2:
                matches.append(items)
    return np.array(matches)

def load_data(data_dir, split):
    
    mbz_eid2id = load_openKE_dict(data_dir, 'mbz')
    wd_eid2id = load_openKE_dict(data_dir, 'wd')
    
    eid_matches = load_matches(data_dir, split)
    matches = convert_matches(eid_matches, mbz_eid2id, wd_eid2id)
    
    return matches

def load_entity_embeddings(embedding_fpath):
    ''' Loads embeddings from fpath and returns entity embeddings. 
    '''
    embeddings = json.load(open(embedding_fpath))
    return np.array(embeddings['ent_embeddings.weight'])

class EMDataset(data.Dataset):
    ''' Defines an entity matching dataset. Loads matches from file and matches 
    them to embeddings via index dictionaries. 
    ''' 

    def __init__(self, data_dir, split):
        ''' Initializes dataset 

            Args: 
                data_dir: path to directory for matched pairs
                mbz_fpath: path to MBZ embeddings 
                wd_fpath: path to WD embeddings 
        ''' 
        self.matches = load_data(data_dir, split)
    
    def __len__(self):
            'Denotes the total number of samples'
            return len(self.matches)

    def __getitem__(self, index):
            'Generates one sample of data'
            
            x = np.array(self.matches[index])
            y = np.array([1.0])
            return x,y 


def get_negative_samples(pairs, neg_per_sample = 1):
    ''' Generate a set of negative samples for passed positive matches. 
    
        Args: 
            pairs: pairs of positive matches 
            neg_per_sample: number of negatives to generate per positive match 
        
        Returns: 
            x: array of negative pairs 
            y: negative labels 
    ''' 
    mbz_ids = pairs[:, 0]
    wd_ids  = pairs[:, 1]

    neg_mbz = []
    neg_wd = []
    for i in range(neg_per_sample):
        neg_mbz.extend(mbz_ids)
        neg_wd.extend(np.roll(wd_ids, i).tolist())

    x = np.zeros((len(mbz_ids)*neg_per_sample, 2))
    x[:, 0] = neg_mbz 
    x[:, 1] = neg_wd
    y = [[0]]*len(x)
    return torch.LongTensor(x), torch.DoubleTensor(y)

def get_loaders(data_dir, train_batch_size = 16, shuffle = True):

    # train dataset 
    train_data = EMDataset(data_dir, 'train')
    train_loader = data.DataLoader(train_data, batch_size = train_batch_size, shuffle = shuffle)

    # validation dataset 
    val_data = EMDataset(data_dir, 'val')
    val_loader = data.DataLoader(val_data, batch_size = 1, shuffle = shuffle)

    # test dataset
    test_data = EMDataset(data_dir, 'test')
    test_loader = data.DataLoader(test_data, batch_size = 1, shuffle = shuffle)

    return train_loader, val_loader, test_loader

def load_candidates(data_dir):

    wd_eid2id = load_openKE_dict(data_dir, 'wd')

    fpath = os.path.join(data_dir, 'candidate_tails.txt')
    candidates = []
    with open(fpath, 'r') as in_file:
        for line in in_file:
            candidates.append(wd_eid2id[line.strip()])
    return np.array(candidates)
    
    
def load_etypes(data_dir):
    mbz_eid2index = load_openKE_dict(data_dir, 'mbz')

    mbz_index2type = {}
    fpath = os.path.join(data_dir, 'id2type.txt')
    matches = []
    with open(fpath, 'r') as in_file:
        for line in in_file:
            eid, etype = line.strip().split("\t")
            if eid in mbz_eid2index:
                index = mbz_eid2index[eid]
                mbz_index2type[index] = etype
    return mbz_index2type