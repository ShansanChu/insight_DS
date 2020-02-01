import pandas
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import sys
import numpy as np
from numpy import save
from scipy.spatial.distance import cosine
sys.path.append("./data")
def preprocessing(df):
    """
    Transform the csv data to one corpus
    """
    #df=pandas.read_csv(file)
    df=df[['content','title','date']]
    df=df[df.content.str.len()>100]
    df=df[~df.title.isin(['Featured Cannabis Companies','Alan Brochstein, CFA - Media Mentions'])]   #exclude some not useful data
    df=df.drop_duplicates(subset='title',keep='first')
    df.title=df.title.apply(lambda x:[x])  #every title suppose to be one sentence
    #corpus=df.title.str.cat(sep='. ')
    corpus=df.title.agg(sum)
    return corpus,df


def word_embedding(df):
    """
    embedding sentences with sentence_transformer
    input:pandas dataframe all sentences to cluster
    in order to save time, store embedding vectors

    """
    # 'bert-base-nli-stsb-mean-tokens' is one of the pretrained model
    # Other model options include: 'bert-base-nli-mean-tokens','bert-large-nli-mean-tokens','bert-large-nli-stsb-mean-tokens'
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    df['vec'] = df['title'].apply(lambda x: embedder.encode([str(x[0])]))
    corpus_emb = df.vec.agg(sum)
    save('embedding.npy',corpus_emb)   #save the data for late usage rather than calculating every time
    # corpus_emb=embedder.encode(sent_tokenize(corpus))
    #return corpus_emb

if __name__=='main':
    summary = pandas.read_csv('summary.csv')
    word_embedding(summary)
