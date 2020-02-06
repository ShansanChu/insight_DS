import pandas
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import sys
import numpy as np
from numpy import save
import os
package_dir = os.path.dirname(os.path.abspath('__file__'))
thefile = os.path.join(package_dir, '../data/summary.csv')


def word_embedding(df):
    """
    embedding sentences with sentence_transformer
    input:pandas dataframe all sentences to cluster
    in order to save time, store embedding vectors

    """
    # 'bert-base-nli-stsb-mean-tokens' is one of the pretrained model
    # Other model options include: 'bert-base-nli-mean-tokens','bert-large-nli-mean-tokens','bert-large-nli-stsb-mean-tokens'
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    df['vec'] = df['title'].apply(lambda x: embedder.encode([str(x)]))
    corpus_emb = df.vec.agg(sum)
    save('embedder1.npy', corpus_emb)   # save the data for late usage rather than calculating every time
    # corpus_emb=embedder.encode(sent_tokenize(corpus))
    # return corpus_emb


if __name__ == '__main__':
    summary = pandas.read_csv(thefile)
    print('summary loaded')
    word_embedding(summary)
    print('embedding done')
