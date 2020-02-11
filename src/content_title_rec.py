"""
Find close based on tile and summary of articles
"""
import pandas
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import sys
import numpy as np
import re
from numpy import save
from scipy.spatial.distance import cosine


def get_maps(summary):
    corpus_map=summary.sent_len.tolist()
    end=0
    maps={}
    for idx,i in enumerate(corpus_map):
        for j in range(i):
            maps[j+end+1]=idx
        end+=i
    return maps


def print_con_close_docs(doc_em,corpus_emb,df,maps):
    """
    print the top 10 sentences close to specific doc
    """
    fs_dis=[cosine(corpus_emb[i],doc_em) for i in range(len(corpus_emb))]
    fs_0=np.argsort(np.array(fs_dis))
    leng=df.shape[0]
    arts=[]
    distances=[]
    id=0
    res = []
    date = []
    summ = []
    title = []
    org = []
    while True:
        i=fs_0[id]
        if i>=leng:
            idx=maps[i-leng+1]
        else:
            idx=i
        if idx not in arts:
            arts.append(idx)
            distances.append(round(fs_dis[i],3))
        id+=1
        if len(arts)>20:
            break
    for i in arts:
        title.append(df.title.iloc[i])
        summ.append(df.summ.iloc[i])
        date.append(df.date.iloc[i])
        all_orgs = [re.sub('\W+', ' ', i) for i in df['ORG'].iloc[i][1:-1].split(',')]
        if len(all_orgs[0]) > 1:
            org.append(', '.join(i for i in set(all_orgs)))
        else:
            org.append('None')
        #org.append(df.ORG.iloc[i])
        #print(cor[i])
        #if i>leng:
            #i=maps[i-leng+1]
            #print('Baaed on content')
        res.append((df.title.iloc[i],"----",df.summ.iloc[i]))
        #print([fs_dis[i] for i in fs_0[:10]])
    d = {'Date': date, 'Title': title, 'Distance': distances, 'Organization': org}
    return res,pandas.DataFrame(d)


def get_con_close_docs(text, df):
    # text='SEC Charges Hemp Inc. and CEO Bruce Perlowin with Fraud.'
    # corpus = get_corpus(df)
    maps=get_maps(df)
    df=df.drop(columns=['sents'])
    #df['sents'] = df['summ'].apply(lambda x: sent_tokenize(str(x)))
    #corpus_all = df['title'].apply(lambda x: [str(x)]).agg(sum) + df['sents'].agg(sum)
    embedder = SentenceTransformer('bert-base-nli-mean-tokens') # embedder's length is not consistent with embedding, don't use it
    doc_em = embedder.encode([text])
    # embedding = word_embedding(corpus) # it takes a long time to do word embedding, save pre-embedding one
    embedding = np.load('all_emb.npy')
    embedding = list(embedding)
    # print(len(embedding),'---------',len(corpus_all),'-----',len(df['sents'].agg(sum)))
    # print(len(corpus_all))
    return print_con_close_docs(doc_em, embedding, df,maps)