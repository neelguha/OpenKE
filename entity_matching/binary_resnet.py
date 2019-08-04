# -*- coding: utf-8 -*-
import torch
from torch import nn
from torchvision import models 
import numpy as np

class BinResNet(nn.Module):
    def __init__(self, mbz_embed, wd_embed, freeze_resnet = False):
        """ Learn a model on top of the cross product of embeddings. 

        Args: 
            mbz_embeddings: matrix of MBZ entity embeddings 
            wd_embeddings: matrix of WD entity embeddings
            freeze_resnet: whether to update the resnet weights during training 
        
        """
        super(BinResNet, self).__init__()
        self.mbz_dim = len(mbz_embed[0])
        self.wd_dim = len(wd_embed[0])

        self.mbz_embeddings = nn.Embedding.from_pretrained(torch.Tensor(mbz_embed), freeze=True)
        self.wd_embeddings = nn.Embedding.from_pretrained(torch.Tensor(wd_embed), freeze=True)
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, 3, stride=1, padding=0, dilation=1, groups=1, bias=True),
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )




    def forward(self, x):
        """
            Args:
                x: a list of tuples (mbz, wd) corresponding to matching mbz and wd entity indices. 

            Returns:
                y_pred: an array of scores, corresponding to positive matches x and negative pairs x'
        """

        # generate negative samples by offsetting wd in x by one 
        mbz_sample = torch.LongTensor(x[:, 0])
        wd_sample = torch.LongTensor(x[:, 1])
        num_samples = len(mbz_sample)

        mbz_embed = self.mbz_embeddings(torch.LongTensor(mbz_sample)).view(num_samples, self.mbz_dim)
        wd_embed = self.wd_embeddings(torch.LongTensor(wd_sample)).view(num_samples, self.wd_dim)
        

        prod  = torch.einsum('bi,bj->bij', (mbz_embed, wd_embed)).view(-1, 1, self.mbz_dim, self.wd_dim)
        x = self.features(prod.float())
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def predict_matches(self, mbz_id, wd_candidates):
        ''' Returns the ID and score of the predicted matching entity

            Args: 
                mbz_id: musicbrainz_id to match 
                wd_candidates: list of wikidata entity IDs to search for match 
        '''
        mbz_dup = [mbz_id]*len(wd_candidates)
        data = np.zeros((len(mbz_dup), 2))
        data[:, 0] = mbz_dup
        data[:, 1] = wd_candidates 

        scores = self.forward(data)
        best_index = scores[:, 1].max(0)[1]
        return wd_candidates[best_index], scores[best_index, 1]