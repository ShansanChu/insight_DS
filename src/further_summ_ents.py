"""
To further summarize the articles and search both titles and contents
"""
import pandas
import warnings
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from summarizer import Summarizer #pretrained model to extract important sentences
from nltk.tokenize import sent_tokenize
import spacy
import os
nlp = spacy.load("en_core_web_lg")
model=Summarizer()

package_dir = os.path.dirname(os.path.abspath('__file__'))
thefile = os.path.join(package_dir, '../data/sum_ents.csv')
npy_file=os.path.join(package_dir,'../all_emb.npy')
further_csv_file=os.path.join(package_dir,'../data/further_summ_ents.csv')


def remove_unrelated(string):
    """
    further clean and remove unrelated content
    """
    unr=['To learn more','For more information','Contact:','Original press','Further information','About']
    for i in unr:
        idx=string.find(i)
        if idx!=-1:
            string=string[:idx]
    #string=re.sub('\W+',' ',string)
    string=re.sub(r'www.\S+', '',string)#remove url
    string=re.sub(r'[\w\.-]+@[\w\.-]+','', string)#remove email
    return string


def remove_further(string):
    string=re.sub(r'\(.*?\)','',string) #remove letter in parenthesis
    string=re.sub(r'\/.*?\/*–','.',string)
    string=re.sub(r'Inc.','Inc',string)
    return string


summary = pandas.read_csv(thefile)
summary = summary.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
summary.content = summary.content.apply(lambda x: remove_unrelated(str(x)))
summary.content = summary.content.apply(lambda x: remove_further(str(x)))
summary['summ']=summary.content.apply(lambda x: model(str(x),min_length=100))
embedder=SentenceTransformer('bert-base-nli-mean-tokens')
summary['vec'] = summary['title'].apply(lambda x: embedder.encode([str(x)]))
summary['summ_vec']=summary['summ'].apply(lambda x: embedder.encode(sent_tokenize(str(x))))
summary['sents']=summary['summ'].apply(lambda x: sent_tokenize(str(x)))
summary['sent_len']=summary.summ.apply(lambda x:len(sent_tokenize(str(x))))
summary['ents']=summary['summ'].apply(lambda x:nlp(x).ents)
summary['ORG']=summary['ents'].apply(lambda x:[i.text for i in x if i.label_=='ORG'])
orgs = ['Tweed', 'Aurora', 'GW', 'Canopy Growth', 'Aphria', 'FDA', 'Health Canada', 'Shoppers Drug Mart']
for org in orgs:
    summary[org] = summary['ORG'].apply(lambda x: 1 if org in '. '.join(i for i in x) else 0)
summary['GPE'] = summary.ents.apply(lambda x: [i.text for i in x if i.label_ == 'GPE'])
summary['Ca'] = summary['GPE'].apply(lambda x: 1 if 'Canada' in x or 'Ontario' in x or 'Alberta' in x or 'British Columbia' in x else 0)
title_emb = summary.vec.agg(sum)
summ_embedding = summary.summ_vec.agg(sum)
all_embedding = title_emb+summ_embedding
np.save(npy_file,all_embedding)
summary=summary.drop(columns=['summary'])
summary.to_csv(further_csv_file)