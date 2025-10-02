"""Simple Tokeniser class implemented """

import re
from pathlib import Path
from typing import Dict,List,Optional,Union
import logging


#logging setup
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)




class Tokenizer:

    def __init__(self):
       
        self.fwd_map:Dict[str,str]={}
        self.re_map:Dict[str:str]={}
        self.encoded:str=""
        self.decoded:str=""
        self.unk_token:str="<UNK>"
        self.EOT:str="<END>"
        self.BOS="<BOS>"
        self.space:str="<SPACE>"
        self.patterns :str= r'''
        \w+(?:'\w+)? |    # words with optional apostrophes
        \d+\.?\d* |       # numbers with optional decimals
        \s+ |             # whitespace
        [^\w\s]           # punctuation and special chars
        '''
    
    def _tokenize(self,text:str)->List[str]:
        regex_pattern=r'''([\w']+|[!?.,;:"'\-()\[\]{}<>/]|\s+)'''
        tokens=re.findall(regex_pattern,text)
        return [token for token in tokens if token.strip() or token==' ']
    
    def _build_vocab(self,tokens:List[str])->None:

        vocab=sorted(set(tokens))
        vocab.extend([self.unk_token,self.EOT,self.space,self.BOS])
        self.fwd_map={token : str(i) for i,token in enumerate(vocab)}
        self.re_map={str(i):token for i,token in enumerate(vocab)}

       

        logger.info(f"Vocabulary is built with {len(self.fwd_map)} tokens")


    def encode(self,raw_text,src_file_path=None) -> List[str]:


        try :
            src_file_path=Path(src_file_path)
            if not src_file_path.exists():
                raise FileNotFoundError(f"Check the path once again")
            src_text=src_file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Error reading file {src_file_path} : {e}")
            raise

        # Build the vocab using the verdict data
        vocab_tokens=self._tokenize(src_text)
        self._build_vocab(vocab_tokens)

        #Tokenise the given text
        tokens=self._tokenize(raw_text)
        print(tokens)
        for token in tokens:

            try:
                if token.strip():
                    if len(self.encoded)==0:
                        self.encoded+=self.fwd_map[self.BOS]+" "
                    
                    if token not in self.fwd_map.keys():
                        self.encoded+=self.fwd_map[self.unk_token]+" "
                    else:
                        self.encoded+=self.fwd_map[token]+" "
                else:
                    self.encoded+=self.fwd_map[self.space]+" "
            except Exception as e:
                logger.error(f"Error with token {token}")
                raise
            
        return self.encoded
                 
        # tokens=self._tokenize(text)
        # build_vocab=self._build_vocab(tokens)

        # text_tokens=re.findall(self.patterns,text,re.VERBOSE)
        # # text_split=re.split(r'([,./"<>?_(){][}]|--|\s)',text)
        # text_split=[token for token in text_tokens if token.strip() or token==" "] 
        # text_split_set=sorted(set(text_split))
        # text_split_set.extend([self.unk_token,self.EOT,self.space])

        # self.fwd_map={val:str(i) for i,val in enumerate(text_split_set)}
        
        # print(self.fwd_map)
        # text_split.append("sumootavarisara")
        # for word in text_split:
        #     if word.strip():
        #         if word not in self.fwd_map.keys():
        #             self.encoded+=str(self.fwd_map[self.unk_token])+" "
        #         else:
        #             self.encoded+=str(self.fwd_map[word])+" "
        #     else:
        #         self.encoded+=str(self.fwd_map[self.space])
        # self.encoded+=str(self.fwd_map[self.EOT])
        # return self.encoded

    def decode(self,encoded_txt:str):

        if len(encoded_txt)<1:
            logger.error("Something wrong with encoded text")
            return

        ind_tokens=encoded_txt.split(" ")
        print(self.fwd_map)
        print(self.re_map)
        print(ind_tokens)
        for token in ind_tokens:
            print(type(token))
            try:
                if token.strip():
                    self.decoded+=self.re_map[token]+" "
            except Exception as e:
                logger.error(f"Something is going wrong with {token} \n")
                raise
        return self.decoded   
        
        # if encoded_txt is not None:
        #     encoded_txt=encoded_txt.split()
        # else:
        #     encoded_txt=""
        #     input_text=raw_text
        #     for word in input_text.split(" "):
        #         if word not in self.fwd_map.keys():
        #             encoded_txt+=self.fwd_map[self.unk_token] +" "
        #         else:
        #             encoded_txt+=self.fwd_map[word] + " "
        # print("ENCODED TXT ",encoded_txt)
        # print(self.re_map)
        
        # for token in encoded_txt.split(" "):
        #     try:
        #         if self.re_map[token] is not None :
        #             self.decoded+=self.re_map[token] + " "
             
        #     except KeyError:
        #         print(f"Key not there for {token}")
        
        return self.decoded


tokeniser=Tokenizer()
encoding=tokeniser.encode(raw_text="a b c d e f ur mom and your sister and your broke ass car ",src_file_path="the-verdict.txt")
decoding=tokeniser.decode(encoding)
with open("simple_tokeniser_out.txt","w") as f:
    f.write(decoding)
print("ENCODED  "  , type(encoding),"\n")
print("DECODED  ",decoding)




