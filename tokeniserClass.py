"""Simple Tokeniser class implemented """
import re

class Tokenizer:

    def __init__(self):
        self.vocabulary={}
        self.fwd_map={}
        self.re_map={}
        self.encoded=""
        self.decoded=""

    def encode(self,file_path):

        with open(file_path,"r") as f:

            text = f.read()
            # text=text[0]

        # Using  regex to split
        text_split=re.split(r'([,./"<>?_(){][}]|--|\s)',text)
        text_split_set=sorted(set(text_split))

        self.fwd_map={val:str(i) for i,val in enumerate(text_split_set)}

        for word in text_split:
            if word.strip():
                self.encoded+=str(self.fwd_map[word])+" "
        
        return self.encoded

    def decode(self,encoded_txt):
        self.re_map={v:k for k,v in self.fwd_map.items()}
        
        encoded_txt=encoded_txt.split()
        print(encoded_txt)
        
        for token in encoded_txt:
            try:
                if self.re_map[token] is not None:
                    self.decoded+=self.re_map[token] + " "
            except KeyError:
                print(f"Key not there for {token}")
        
        return self.decoded


tokeniser=Tokenizer()
encoding=tokeniser.encode("the-verdict.txt")
decoding=tokeniser.decode(encoding)
print("ENCODED  "  , type(encoding),"\n")
print("DECODED  ",decoding)




