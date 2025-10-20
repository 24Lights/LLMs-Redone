import re
import logging
from pathlib import Path
from typing import Dict,List,Optional,Union
import importlib
import tiktoken

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class BytePairEncoder:

    """BPE Encoder implementation"""
    """Made in 1994 - The most common pair of consecutive bytes of data is replaced with a byte that doesnt occur
    in data"""

    def __init__(self):
        self.tokeniser=tiktoken.get_encoding("gpt2")
        self.gpt3tokeniser=tiktoken.get_encoding("p50k_base")
        self.gpt4tokeniser=tiktoken.get_encoding("cl100k_base")
    
    def encoder(self,text:str):


        tokens=self.tokeniser.encode(text,allowed_special={"<|endoftext|>"})

        return tokens
    
    def decoder(self,encoded_texts:List[int]):

        decoded=self.tokeniser.decode(encoded_texts)

        return decoded

    

bpe=BytePairEncoder()
encoded=bpe.encoder(text="alpha beta gama detla phi si quonki thatwhy .")
decoded=bpe.decoder(encoded_texts=encoded)
print(encoded)
print(decoded)
