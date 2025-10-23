import torch
import tiktoken
import numpy as np
from torch.utils.data import DataLoader,Dataset
import logging
from itertools import islice

"""
1. DATASET MAKING
2. DATALOADER
3. TOKEN EMBEDDINGS 
4. POSITIONAL EMBEDDINGS
5. MULTI HEAD MASKED ATTENTION
    5A. INITIALISE K,Q,V
    5B. Q*K.T
    5C. DIVIDE BY SQRT N_DIMS
    5D. APPLY MASK
    5E. APPLY SOFTMAX
    5F. * THE RESULTANT BY V
    5G. DROPOUT
6. APPLY LAYER NORM
7. PASS THROUGH FFN 
8. APPLY GELU
9. ADD THE RESULTANT WITH SKIP CONNECTION
10. REPEAT THIS BLOCK 12 TIMES FOR GPT 2 MODEL

"""

######-----------------SETUP-BEGIN-----------------#####


torch.manual_seed(42)
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
######-----------------SETUP-END-----------------#####



"""
1. DATASET MAKING
"""


class DS(torch.nn.Module):

    def __init__(self,path=None,context_sz=256,stride=64):
        super().__init__()
        if not path is None:
            with open(path,"r") as f:
                self.raw_data=f.read()
        else:
            logger.error("FILE PATH IS NOT GIVEN")
        
        # BYTE PAIR ENCODING
        self.encoder=tiktoken.get_encoding("cl100k_base")
        self.encoded_text=self.encoder.encode(self.raw_data,allowed_special={"|<endoftext>|"})
        self.inputs=[]
        self.outputs=[]
        self.IO_PAIRS=[]
        self.IO_PAIR_MAKER(self.encoded_text,context_sz=context_sz,stride=stride)

    
    def IO_PAIR_MAKER(self,encoded_text,context_sz,stride):

        for i in range(0,len(encoded_text)-context_sz,stride):

            input_token=encoded_text[i:i+context_sz]
            output_token=encoded_text[i+1:i+1+context_sz]

            self.inputs.append(input_token)
            self.outputs.append(output_token)
            self.IO_PAIRS.append((input_token,output_token))

    def __getitem__(self,idx):

        return torch.tensor(self.inputs[idx]),torch.tensor(self.outputs[idx])
    
    def __len__(self):

        return len(self.inputs)

    def get_vocab_sz(self):

        return self.encoder.n_vocab

class DL(torch.nn.Module):

    def __init__(self,batch_size,dataset):
        super().__init__()

        self.dataloader=DataLoader(dataset,batch_size,shuffle=True)
        self.iterator=iter(self.dataloader)
    
    def get_loader(self):

        return self.dataloader
    
    def get_batches(self,num_batches):

        return list(islice(self.dataloader,num_batches))

class Token_Embeddings(torch.nn.Module):

    def __init__(self,vocab_size,emb_dim):
        super().__init__()
        self.embeddings=torch.nn.Embedding(vocab_size,emb_dim)
    
    def forward(self,inputs):

        return self.embeddings(inputs)

class Positional_Embeddings(torch.nn.Module):

    def __init__(self,context_sz,emb_dim):
        super().__init__()
        self.context_sz=context_sz
        self.emb_dim=emb_dim

        self.pos_embeddings=torch.nn.Embedding(context_sz,emb_dim)

    def forward(self,x):

        b,context_sz,emb_dim=x.size()
      
        indices=torch.arange(0,self.context_sz)
        self.pos_embeds=self.pos_embeddings(indices)

        return x+self.pos_embeds    

class LayerNorm(torch.nn.Module):

    def __init__(self,emb_dim):
        super().__init__()
        self.layer_norm=torch.nn.LayerNorm(emb_dim,eps=1e-6)

    def forward(self,x):

        return self.layer_norm(x)         

class MHA(torch.nn.Module):

    def __init__(self,d_in,d_out,kqv_bias=False,num_heads=4,context_len=6):

        super().__init__()
        self.num_heads=num_heads

        assert(d_out%self.num_heads ==0)

        self.d_out=d_out
        self.head_dim=d_out//self.num_heads
        self.W_keys    = torch.nn.Linear(d_in,d_out,kqv_bias)
        self.W_queries = torch.nn.Linear(d_in,d_out,kqv_bias)
        self.W_values  = torch.nn.Linear(d_in,d_out,kqv_bias)

        self.dropout   = torch.nn.Dropout(0.5)

        self.register_buffer("maskUPD",torch.triu(torch.ones(context_len,context_len),diagonal=1))
    
    def mask(self,attn_score_matrix):

        #attention score matrix shape is b,num_heads,context_sz,context_Sz
        
        dim=attn_score_matrix.shape[-1]

        mask=torch.tril(torch.ones(dim,dim),diagonal=0)
        filled_mask=mask.masked_fill(mask==0,-torch.inf)

        return filled_mask
    
    def forward(self,x):

        b,context_sz,emb_dim=x.size()
        

        
        keys    =self.W_keys(x)
        queries =self.W_queries(x)
        values  =self.W_values(x) # shape will be b,context_sz,d_out
        
        keys    =keys.view(b,context_sz,self.num_heads,self.head_dim)
        queries =queries.view(b,context_sz,self.num_heads,self.head_dim)
        values  =values.view(b,context_sz,self.num_heads,self.head_dim)

        # group by num_heads

        keys    = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values  = values.transpose(1,2) # shape will be b,num_heads,context_sz,head_dim


        attention_scores = queries @ keys.transpose(2,3)
        scaled_attention_scores= attention_scores/torch.sqrt(torch.tensor(self.head_dim))

        mask=self.maskUPD[:context_sz,:context_sz].unsqueeze(0).unsqueeze(0) # shape will be 1,1,contx_sz,contx_sz
        masked_attn = scaled_attention_scores.masked_fill(mask.bool(),-torch.inf)

        normalised_masked_attn=torch.softmax(masked_attn,dim=-1) # shape will be b,num_heads,context_sz,context_Sz

        kqv_matrix= normalised_masked_attn @ values # shape will be b,num_heads,context_sz,head_dim

        dropped_kqv=self.dropout(kqv_matrix)
        dropped_kqv=dropped_kqv.transpose(1,2)

        dropped_kqv=dropped_kqv.contiguous().view(b,context_sz,self.d_out)

        
        return dropped_kqv # shape will be b,context_sz,d_out
        
class FFN(torch.nn.Module):

    def __init__(self,n_layers,emb_dim):

        super().__init__()

        self.layer_sizes=[emb_dim] + [4*emb_dim]*(n_layers-1) + [emb_dim]

                
        self.layer_list=[]

        for i in range(n_layers):

            self.layer_list.append(torch.nn.Sequential(
                torch.nn.Linear(self.layer_sizes[i],self.layer_sizes[i+1]),
                torch.nn.GELU()))


        self.layers=torch.nn.ModuleList(self.layer_list)
  

    def forward(self,x):

        b,context_sz,emb_dim=x.size() # emb_dim or d_out , both are same here
       
        for layer in self.layers:


            layer_out=layer(x)
            x=layer_out 
        
        # after all layers , shape will be b, context_sz, 128 based on the init configs
        
        return x

class SkipConnection(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
    
    def forward(self,latest_matrix,old_matrix):

        self.a=latest_matrix # shape is b,context_sz,emb_dim or d_out
        self.b=old_matrix    # shape is b,context_sz,emb_dim
        
        res=self.a+self.b

        return res

class TransformerBlock(torch.nn.Module):

    def __init__(self,emb_dim):
        super().__init__()
        """
        2 different instances of layer norm because the gamma and beta parameters in 
        shift and scale parameters are trainable. thats why we use different instances as the mha 
        inputs and ffn inputs are different distributions.
        """
        self.layer_norm1=LayerNorm(emb_dim)
        self.layer_norm2=LayerNorm(emb_dim)
        self.mha=MHA(emb_dim,emb_dim,True)
        self.skip=SkipConnection()
        self.ffn=FFN(3,emb_dim)
        self.dropout=torch.nn.Dropout(0.4)
        
    
    def forward(self,x):

        # x shape is b,context_sz,emb_dim

        # Normalise the layer
        first_LN=self.layer_norm1(x) # shape now is b,contetx_sz,emb_dim

        # feed this LN to MHA
        mha_out=self.mha(first_LN) # shape now is b,context_sz,emb_dim

        # # normalise MHA outs
        # second_LN=self.layer_norm2.forward(mha_out) # shape now is b,context_sz,emb_dim

        # skip connextion addition    
        skip_out_1=self.skip(mha_out,x) # shape now is b,context_sz,emb_dim

        # normalise the skip_out
        second_LN=self.layer_norm2(skip_out_1) # shape now is b,context_sz,emb_dim

        # pass the out to FFN 
        ffn_out_1=self.ffn(second_LN) # shape now is b,context_sz,emb_dim

        # skip connection addition
        skip_out_2=self.skip(ffn_out_1,skip_out_1)

        return self.dropout(skip_out_2)

class Preprocessor(torch.nn.Module):

    def __init__(self,path):
        super().__init__()

        self.dataset=DS(path,6,64)
        self.dataloader=DL(16,self.dataset)
        self.vocab_sz=self.dataset.get_vocab_sz()
        self.token_embedding=Token_Embeddings(self.vocab_sz,256)
        self.position_embedding=Positional_Embeddings(6,256)



class Processor(torch.nn.Module):

    def __init__(self,path):

        super().__init__()
        self.preprocessor=Preprocessor(path)
        self.transformer=TransformerBlock(256)

    def forward(self):

        for i,(batch_in,batch_out) in enumerate(self.preprocessor.dataloader.get_loader()):

            batch_in=batch_in.long()
            token_embeds=self.preprocessor.token_embedding(batch_in)
            pos_embeds=self.preprocessor.position_embedding(token_embeds)

            trans_out_1=self.transformer(pos_embeds)

            print(f"Batch {i} --> INPUT Shape {batch_in.shape}")
            print(f"Batch {i} --> TRANS_IN Shape {pos_embeds.shape}")
            print(f"Batch {i} --> TRANS_OUT Shape {trans_out_1.shape}")


if __name__ == "__main__":


    proc=Processor("/home/ganeshka/Desktop/LLMs Redone/the-verdict.txt")
    proc.forward()










        

            


        

        



        
        



    








    









    
            
            
            
        
    

