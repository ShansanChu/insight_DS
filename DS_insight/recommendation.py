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
    output:embedding vectors

    """
    # 'bert-base-nli-stsb-mean-tokens' is one of the pretrained model
    # Other model options include: 'bert-base-nli-mean-tokens','bert-large-nli-mean-tokens','bert-large-nli-stsb-mean-tokens'
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    df['vec'] = df['title'].apply(lambda x: embedder.encode([str(x[0])]))
    corpus_emb = df.vec.agg(sum)
    save('embedding.npy',corpus_emb)   #save the data for late usage rather than calculating every time
    # corpus_emb=embedder.encode(sent_tokenize(corpus))
    #return corpus_emb

def print_close_docs(doc_em,corpus_emb,corpus,df):
    """
    print the top 10 sentences close to specific doc
    """
    fs_dis=[cosine(corpus_emb[i],doc_em) for i in range(len(corpus_emb))]
    fs_0=np.argsort(np.array(fs_dis))
    res=[]
    for i in fs_0[:10]:
        res.append(df.date.iloc[i]+"---------"+corpus[i]+'========:'+str(fs_dis[i]))
        #print([fs_dis[i] for i in fs_0[:10]])
    return res
def get_close_docs(text,df):
    #text='SEC Charges Hemp Inc. and CEO Bruce Perlowin with Fraud.'
    corpus, df = preprocessing(df)
    embedder=SentenceTransformer('bert-base-nli-mean-tokens')
    doc_em=embedder.encode(sent_tokenize(text))
    #embedding = word_embedding(corpus) # it takes a long time to do word embedding, save pre-embedding one
    embedding=np.load('embedder.npy')
    embedding=list(embedding)
    return print_close_docs(doc_em,embedding,corpus,df)
