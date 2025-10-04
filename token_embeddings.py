import numpy as np
import re
import tiktoken
import logging
import torch
from torch.utils.data import Dataset,DataLoader

import gensim.downloader as api

model=api.load("word2vec-google-news-300")

# Setup logger
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


"""Token embeddings done from scratch"""

word_vectors=model

print(word_vectors['alpha'].shape)

# similar words : This does king + lady - royal
print(word_vectors.most_similar(positive=['king','lady'],negative=['royal'],topn=15))

# similarity
print(word_vectors.similarity('lion','dog'))
print(word_vectors)


"""Making embeddings from scratch"""

class EmbeddingMaker():

    def __init__(self,vocab_sz:int=100,emb_dim:int=5):

        torch.manual_seed(42)
        self.embedding=torch.nn.Embedding(vocab_sz,emb_dim)

    def get_embeddings(self):

        return self.embedding

emb=EmbeddingMaker()
embedding_table=emb.get_embeddings()

input_ids=torch.tensor([1,2,3,4,5])
print(embedding_table(input_ids))