"""
Find close based on tile
"""
import pandas
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import sys
import numpy as np
import re
from numpy import save
from scipy.spatial.distance import cosine
sys.path.append("./data")


def get_corpus(df):
    """
    Transform the csv data to one corpus
    """
    df.title=df.title.apply(lambda x: [x])  # every title suppose to be one sentence
    # corpus=df.title.str.cat(sep='. ')
    corpus=df.title.agg(sum)
    return corpus


def print_close_docs(doc_em, corpus_emb, df):
    """
    print the top 10 sentences close to specific doc
    """
    fs_dis = [cosine(corpus_emb[i], doc_em) for i in range(len(corpus_emb))]
    fs_0 = np.argsort(np.array(fs_dis))
    res = []
    date = []
    distance = []
    title = []
    org = []
    for i in fs_0[:20]:
        # print(df.ORG.iloc[i][1:-1])
        date.append(df.date.iloc[i])
        distance.append(round(fs_dis[i],3))
        title.append(df.title.iloc[i])
        res.append(df.date.iloc[i]+"---------"+df.title.iloc[i]+'========Distance is: '+str(round(fs_dis[i],3)))
        all_orgs = [re.sub('\W+', ' ', i) for i in df['ORG'].iloc[i][1:-1].split(',')]
        if len(all_orgs[0]) > 1:
            res.append('The related companies/Organization are:' +', '.join(i for i in set(all_orgs)))
            org.append(', '.join(i for i in set(all_orgs)))
        else:
            res.append('None company found')
            org.append('None')
        # print([fs_dis[i] for i in fs_0[:10]])
    d = {'Date': date, 'Distance': distance, 'Organization': org, 'Title': title}
    return res, pandas.DataFrame(d)


def get_close_docs(text, df):
    # text='SEC Charges Hemp Inc. and CEO Bruce Perlowin with Fraud.'
    # corpus = get_corpus(df)
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    doc_em = embedder.encode([text])
    # embedding = word_embedding(corpus) # it takes a long time to do word embedding, save pre-embedding one
    embedding = np.load('embedder1.npy')
    embedding = list(embedding)
    return print_close_docs(doc_em, embedding, df)
