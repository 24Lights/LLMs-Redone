import re
import os
from pathlib import Path
import logging
import tiktoken
from typing import List,Dict,Union
import torch
from torch.utils.data import Dataset,DataLoader

# logging setup
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)



class DataSet(Dataset):

    """Data loading implementation done"""

    def __init__(self,file_path:str,context_sz=None,stride=3)->None:

        try :
            if not file_path is None:   
                file=Path(file_path)
                data=file.read_text()
                print(len(data))
        except FileNotFoundError as e:
            logger.error(f"File is not found at path {file_path}")
        
        self.tokeniser=tiktoken.get_encoding("gpt2")
        self.stride=stride
        self.context_sz=context_sz
        self.target_pairs_manual=[]
        self.input_ids=[]
        self.output_ids=[]

        encoded_text=self._encode(data)
        self.input_target_pair_maker(encoded_data=encoded_text)


    
    def _encode(self,raw_text:str)->List[int]:

        encoded=self.tokeniser.encode(raw_text,allowed_special={"<|endoftext|>"})

        return encoded

    def _decode(self,encoded_data:List[int])->List[str]:

        decoded=self.tokeniser.decode(encoded_data)

        return decoded

    def input_target_pair_maker(self,encoded_data:List[int]):

        if not encoded_data is None:

            for i in range(1,len(encoded_data)-self.context_sz,self.stride):

                input=encoded_data[i:i+self.context_sz]
                output=[encoded_data[i+self.context_sz]]

                self.target_pairs_manual.extend((input,output))
                self.input_ids.append(torch.tensor(input))
                self.output_ids.append(torch.tensor(output))
                # print(type(input) ,"  ", type(output))
                print(f"{self._decode(input)} ----> {self._decode(output)}")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index],self.output_ids[index]
    


class Dataloader():

    def __init__(self,batch_sz=None,context_sz=4,shuffle=True,drop_last=False,num_workers=5):

        self.dataset=DataSet("the-verdict.txt",context_sz=9,stride=1)

        self.dataloader=DataLoader(self.dataset,3,False,drop_last=True,num_workers=6)

    def get_loader(self):
        
        return self.dataloader


        

data_loader=Dataloader()
loader=data_loader.get_loader()

iterator=iter(loader)

print(next(iterator))
print(next(iterator))
    







        


        



