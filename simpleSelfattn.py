import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import logging
import tiktoken
import gensim.downloader as api
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


logging.basicConfig(level=logging.INFO)
loger=logging.getLogger(__name__)



"""Simplified Self Attention implemented from Scratch"""


class simpleAttnMech():

    def __init__(self):
        torch.seed()
        self.inputs=torch.tensor(np.random.rand(6,3))
        self.attnweights=torch.tensor(np.random.rand(6,3))

        self.words=['your','journey','starts','with','one','step']

        self.x_cords=self.inputs[:,0].numpy()
        self.y_cords=self.inputs[:,1].numpy()
        self.z_cords=self.inputs[:,2].numpy()

        self.query=self.inputs[1]
        self.attn_scores=torch.empty((self.inputs.shape[0],self.inputs.shape[0]))

    def vizualise(self):
        pass

    
    def getAttnscores(self):

        
        self.attn_scores_2=self.inputs @ self.inputs.T
        
        return torch.softmax(self.attn_scores_2,dim=-1)
    
    def getcontextvector(self):

        return self.attn_scores_2@self.inputs


atm=simpleAttnMech()
res=atm.getAttnscores()
res2=atm.getcontextvector()
print(res)
print(res.sum(dim=-1))
print(res2)

