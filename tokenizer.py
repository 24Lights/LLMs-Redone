"""Make a simple tokeniser"""


# Read the text in the file

with open ("the-verdict.txt","r") as f:
    raw_file=f.read()

print(len(raw_file))


# import re library
import re

split_file=re.split(r'(\s)',raw_file)
split_file_2=re.split(r'([,.]|\s)',raw_file)
print(split_file_2)
print(len(split_file_2))


# Remove the white spaces

text_wo_whites=[item for item in split_file_2 if item.strip()]
print(text_wo_whites)

# split the special characters also

revised_txt_file=re.split(r'([,.?:;!"()_-]|--|\s)',raw_file)
revised_txt_file=[item for item in revised_txt_file if item.strip()]
print(len(revised_txt_file))

# STEP 2 : Convert tokens to token IDS

vocabulary=sorted(set(revised_txt_file))
print(len(vocabulary))

vocab_list={word:str(idx) for idx,word in enumerate(vocabulary)}
print(vocab_list)

tokenised_file=""

for word in re.split(r'([!,./?;:"()_-]|--|\s)',raw_file):
    try:
        if word.strip():
         tokenised_file+=str(vocab_list[word])+" "
    except KeyError:
       print("Not in vocab : ",word)

print(tokenised_file)

with open("tokeniser_out.txt","w") as f:
   f.write(tokenised_file)

# STEP 3 : DeTokenisation

decoded=""

with open("tokeniser_out.txt","r") as f:
   token_ids=f.readlines()
   print(len(token_ids))
   token_ids=token_ids[0].split(" ")
   
print(vocab_list)
reverse_vocab={str(v):k for k,v in vocab_list.items()}
print(reverse_vocab)
for token in token_ids:
   try :
      if reverse_vocab[token]!=None:
        decoded+=str(reverse_vocab[token]) + " "
   except KeyError:
      print(f"Key not found for {token}")
print(decoded)


   

