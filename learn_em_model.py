# Learn convolutional entity-matching model from entity embeddings 

import argparse, os, json,  warnings
warnings.filterwarnings('ignore') # to suppress some matplotlib deprecation warnings
import numpy as np

from entity_matching import data_utils, binary_resnet, model_utils
from utils import * 

import torch.optim as optim
from  torch import nn
import torch 

# parser 
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mbz-embed', help='path to MBZ embeddings')
parser.add_argument('-w', '--wd-embed', help='path to WD embeddings')
parser.add_argument('-d', '--data_dir', help='dataset to learn over')
parser.add_argument('-c', '--cpu', action="store_true", help='train on CPU')
parser.add_argument('-o', '--out_dir', help='directory to store results in')
args = parser.parse_args()






def main():
    mbz_emb_fpath = args.mbz_embed 
    wd_emb_fpath = args.wd_embed 
    data_dir  = args.data_dir

    # load datasets and embeddings 
    print("Loading data...", end="")
    train_loader, val_loader, test_loader = data_utils.get_loaders(data_dir)
    mbz_emb = data_utils.load_entity_embeddings(mbz_emb_fpath)
    wd_emb = data_utils.load_entity_embeddings(wd_emb_fpath)

    # load candidate tails 
    candidates = data_utils.load_candidates(data_dir)
    print("done!")

    # initialize model 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = binary_resnet.BinResNet(mbz_emb, wd_emb)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    model_utils.train(model, optimizer, loss, train_loader, val_loader, candidates, 3, device=device, log_interval=1, num_epochs=100)

    out_fpath = os.path.join(args.out_dir, 'model.pt')
    torch.save(model.state_dict(), out_fpath)
 

    
         


if __name__ == '__main__':
    main()
