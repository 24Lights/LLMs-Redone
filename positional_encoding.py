# from token_embeddings import EmbeddingMaker as EM
import re
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import logging
import tiktoken
from pathlib import Path
from typing import Dict,List,Union

import gensim.downloader as api


logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

"""Implemented positional encoding along with dataset maker, dataloader, token embeddings as well"""

class DS(Dataset):

    def __init__(self,context_sz:int=8,stride:int=4,shuffle=False,file_path:str=None):


        self.context_sz=context_sz
        self.stride=stride
        self.shuffle=shuffle

        self.tokeniser=tiktoken.get_encoding("gpt2")

        if not file_path is None:
            file=Path(file_path)
            data=file.read_text()
        
        self.tokens=self.tokeniser.encode(data,allowed_special={"<|endoftext|>"})

        self.input_tokens=[]
        self.output_tokens=[]
        self._IO_pairs(self.tokens)


    def _IO_pairs(self,encoded_data:List[int]):

        for i in range(0,len(encoded_data)-self.context_sz,self.stride):
            input=encoded_data[i:i+self.context_sz]
            output=encoded_data[i+self.context_sz]

            self.input_tokens.append(torch.tensor(input))
            self.output_tokens.append(torch.tensor(output))
    
    def __getitem__(self, index:torch.tensor):
        
        return (self.input_tokens[index],self.output_tokens[index])

    def __len__(self):
        return len(self.input_tokens)
    
    def get_vocabSZ(self):
        return self.tokeniser.n_vocab

class DL():

    def __init__(self):
        dataset=DS(8,4,True,"the-verdict.txt")
        self.vocab_sz=dataset.get_vocabSZ()
        loader=DataLoader(dataset,2,shuffle=False,num_workers=8,drop_last=True)
        self.dl_iterator=iter(loader)

    def get_batches(self,num_batches:int=1 ):

        batches=[]

        for _ in range(num_batches):
            batch=next(self.dl_iterator)
            print(f"batch type is {type(batch)}")
            batches.append(batch)

        return batches

    def vocabu_sz(self):
        return self.vocab_sz


load=DL()
batches=load.get_batches()
print(len(batches))

class TokenEmbeddings():

    def __init__(self):

        self.dataloader=DL()
        self.vocab_siz=self.dataloader.vocabu_sz()
        self.n_dims=256

        self.embedding_matrix=torch.nn.Embedding(self.vocab_siz,self.n_dims)
    
    def get_embeddings(self):
        return self.embedding_matrix

    def embed_data(self):

        

        data_batches=self.dataloader.get_batches(5)
        print("------------")
        print(len(data_batches[0]))
        print(f"Vocabulary size: {self.vocab_siz}")
        print(f"Data batches structure: {len(data_batches[0])}")

        batch_embeddings=[]

        for batch in data_batches:
            print("batch sz is ",len(batch[0]))
            inp=batch[0]
            out=batch[1]

            inp_embedding=self.embedding_matrix(inp)
            batch_embeddings.append(inp_embedding)



        return batch_embeddings

class PositionalEmbeddings():

    def __init__(self,context_sz:int=8,n_dims:int=256):

        self.pos_embeddings_matrix=torch.nn.Embedding(context_sz,n_dims)
        self.indices=torch.arange(0,context_sz,1)

        self.pos_embeddings=self.pos_embeddings_matrix(self.indices)

    def attach_pos_embeddings(self,token_embeddings):
        res=[]
        print("------------------\n","added pos embeddings are : ",len(self.pos_embeddings),"\n")
        for batch in token_embeddings:
            res.append(batch+self.pos_embeddings)

        print(type(token_embeddings),type(self.pos_embeddings))
        return res

        

        

te=TokenEmbeddings()
res=te.embed_data()
pos_emb=PositionalEmbeddings()
positional_embedded=pos_emb.attach_pos_embeddings(res)
print(res[0].shape)
print(positional_embedded[0].shape)
print(positional_embedded[0])
print(res[0])
print(positional_embedded[0]-res[0])





        

        




