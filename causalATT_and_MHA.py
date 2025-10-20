import torch
import torch.nn as nn
import numpy as np
import logging
import tiktoken
import gensim.downloader as api
from torch.utils.data import Dataset,DataLoader


logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

"""
Causal attention implemented from scratch
"""



class DS(Dataset):
    def __init__(self,file_path):

        with open(file_path,"r") as f:

            raw_text=f.read()
           
        
        self.tokeniser=tiktoken.get_encoding("cl100k_base")
        self.tokens=self.tokeniser.encode(raw_text,allowed_special={"<|endoftext|>"}) 
        self.inputs=[]
        self.outputs=[]
        self.IOPairs=[]
        self.pair_maker(self.tokens,6)
    
    def pair_maker(self,tokens,context_sz=8,stride=4):

        for i in range(0,len(tokens)-context_sz,stride):
            self.inputs.append(tokens[i:i+context_sz])
            self.outputs.append(tokens[i+context_sz])
            self.IOPairs.append((torch.tensor(tokens[i:i+context_sz]),torch.tensor(tokens[i+context_sz])))
    
    
    def __len__(self):
        return len(self.IOPairs)

    def __getitem__(self, index):
        return  self.IOPairs[index][0],self.IOPairs[index][1]
   
class DL():

    def __init__(self,dataset):
        
        
        self.loader=DataLoader(dataset=dataset,batch_size=16,shuffle=False,num_workers=0)
        self.iterator=iter(self.loader)

    def get_loader(self):

        return self.loader
    
    def get_batches(self,num_batches):
        res=[]
        for _ in range(num_batches):
            batch=next(self.iterator)
            res.append(batch)
        
        return res

class TokenEmbeddings():

    def __init__(self,vocab_sz,embedding_dim=256):
        self.embeddings=torch.nn.Embedding(vocab_sz,embedding_dim)

    def embed(self,batches):

        res=[]
        for batch in batches:
            inps=batch[0]
            res.append(self.embeddings(inps))

        return res

class PositionalEmbeddings():

    def __init__(self,embeded_tokens):
        self.tokens=embeded_tokens # embedded tokens is a list of batches
        self.context_sz=self.tokens[0].shape[1]
        self.batch_sz=self.tokens[0].shape[0]
        self.n_dims=self.tokens[0].shape[2]
        self.pos_embed_matrix=torch.nn.Embedding(self.context_sz,self.n_dims)
       

    def get_pos_embeddings(self):

        indices=torch.arange(0,self.context_sz,1)
        self.pos_embeds=self.pos_embed_matrix(indices)
        res=[]

        print("self.tokens.shape ",self.tokens[0].shape)
        for batch in self.tokens:

            res.append(batch+self.pos_embeds)
        
        return res


class CausalAttn(nn.Module):

    def __init__(self,d_in,d_out,context_len=3,kqv_bias=False):

        super().__init__()

        self.W_keys=torch.nn.Linear(d_in,d_out,bias=kqv_bias)
        self.W_queries=torch.nn.Linear(d_in,d_out,bias=kqv_bias)
        self.W_values=torch.nn.Linear(d_in,d_out,bias=kqv_bias)
        self.dropout=torch.nn.Dropout(0.5)
    
    def mask(self,kq):
        b,context_len,context_len_dup=kq.shape

        mask=torch.tril(torch.ones(context_len,context_len))
        masked=kq*mask
        masked=masked.masked_fill(mask==0,-torch.inf)
        

        return masked


    def forward(self,x):
        b,context_len,n_dims=x.shape

        keys=self.W_keys(x)
        queries=self.W_queries(x)
        values=self.W_values(x)

        attn_scores=queries @ keys.transpose(1,2)
        print("atten_scores shape : ",attn_scores.shape)
        masked_attn=self.mask(attn_scores)
        scaled_msk_attn=masked_attn/n_dims**0.5
        norm_masked_attn=torch.softmax(scaled_msk_attn,dim=-1)


        droped_attn=self.dropout(norm_masked_attn)
        # print(norm_masked_attn)


        context_vec=droped_attn@values

        return context_vec
 
class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.num_heads=4
        self.heads=nn.ModuleList([CausalAttn(256,1024) for head in range(self.num_heads)])
    
    def forward(self,input):
        
        

        return torch.cat([head(input) for head in self.heads],dim=-1)


class MultiHeadAttention_2(nn.Module):

    def __init__(self,d_in,d_out,context_len,num_heads=4):
        super().__init__()

        assert(d_out%num_heads==0)
        self.num_heads=num_heads
        self.dim_head=d_out//num_heads
        self.d_out=d_out
        self.W_keys=torch.nn.Linear(d_in,d_out,bias=False)
        self.W_queries=torch.nn.Linear(d_in,d_out,bias=False)
        self.W_values=torch.nn.Linear(d_in,d_out,bias=False)
        self.out_proj=torch.nn.Linear(d_out,d_out)
        self.dropout=torch.nn.Dropout(0.45)

        self.register_buffer("mask",torch.triu(torch.ones(context_len,context_len),diagonal=1))


    def forward(self,x):

        batch_sz,context_len,n_dims=x.shape
        num_tokens=context_len

        keys,queries,values=self.W_keys(x),self.W_queries(x),self.W_values(x) 
        # now shape will be batch_sz,context_sz,d_out
        keys=keys.view(batch_sz,context_len,self.num_heads,self.dim_head)
        queries=queries.view(batch_sz,context_len,self.num_heads,self.dim_head)
        values=values.view(batch_sz,context_len,self.num_heads,self.dim_head)
        #now shape will be (batch_sz,context_len,num_heads,head_dim)

        # grouping by number of heads, so shape will be 
        # batch_sz,num_heads,context_len,head_dim
        keys,queries,values=keys.transpose(1,2),queries.transpose(1,2),values.transpose(1,2)

        attention_scores=queries@keys.transpose(2,3)

        # now shape will be batch_sz,num_heads,context_len,context_len

        mask_bool=self.mask.bool()[:context_len,:context_len]
        attention_scores.masked_fill(mask_bool,-torch.inf)

        attention_scores/=keys.shape[-1]**0.5
        attention_wgts=torch.softmax(attention_scores,dim=-1)
        droped_attention_wgts=self.dropout(attention_wgts)

        # now shape is batch_sz,num_heads,context_len,context_len

        context_vec=droped_attention_wgts @ values

        # now shape is batch_sz,num_heads,context_len,head_dim
        context_vec=context_vec.transpose(1,2)

        # Now combine the last 2 dims , so it will match d_out
        context_vec=context_vec.contiguous().view(batch_sz,context_len,self.d_out)
        context_vec=self.out_proj(context_vec)


        return context_vec











## PREPROCESSING CHECKPOINT ##

print("PREPROCESSING CHECKPOINT BEGIN ")
dataset=DS("/home/ganeshka/Desktop/LLMs Redone/the-verdict.txt")
dataloader=DL(dataset)
batches=dataloader.get_batches(2)
print("BATCH SAMPLE :",batches[0])
vocab_sz=dataset.tokeniser.n_vocab
te=TokenEmbeddings(vocab_sz,256)
embedded_tokens=te.embed(batches)
pe=PositionalEmbeddings(embedded_tokens)
position_embedded=pe.get_pos_embeddings()

print("CHECKS : ")
print("NUMBER OF BATCHES :", len(batches))
print("FIRST BATCH SHAPE - inputs:", batches[0][0].shape, "targets:", batches[0][1].shape)
print("NUMBER OF EMBEDDED BATCHES :", len(embedded_tokens))
print("FIRST EMBEDDED BATCH SHAPE : ", embedded_tokens[0].shape)
print("NUMBER OF POSITION EMBEDDED BATCHES :", len(position_embedded))
print("FIRST POSITION EMBEDDED BATCH SHAPE : ", position_embedded[0].shape)

print("PREPROCESSING CHECKPOINT END ","\n")
#####################################33

# CAUSAL ATTENTION BEGIN

print("CAUSAL ATTENTION BEGINS -------------")
d_in,d_out=position_embedded[0].shape[2],512
ca=CausalAttn(d_in,d_out)

for i,embeddings in enumerate(position_embedded):

    context_vectors=ca.forward(embeddings)
    print(f"CONTEXT VECTOR SHAPE of batch {i} :",context_vectors.shape)
        
# MULTI HEAD ATTENTION 
print("MULTI HEAD ATTENTION BEGIN ---------------")

mha=MultiHeadAttention_2(d_in,d_out,6,4)

for i,embeddings in enumerate(position_embedded):

    context_vectors_mha=mha.forward(embeddings)
    print(f"CONTEXT VECTOR SHAPE of batch {i} - MHA :",context_vectors_mha.shape)
    print(context_vectors_mha)








