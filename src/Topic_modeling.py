import re   # for remove non-english words
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine
import pandas
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF
import sys
import os
package_dir = os.path.dirname(os.path.abspath('__file__'))
thefile = os.path.join(package_dir, '../data/cleaned.csv')
lemmatizer=WordNetLemmatizer()
nltk.download('wordnet')
stop_words=stopwords.words('english')
stop_words.extend(['cannabis', 'company', 'marijuana', 'companies'])


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
        print()


def RemoveHTTP(sentence):
    unr = ['To learn more', 'For more information', 'Contact:', 'Original press', 'Further information']
    for i in unr:
        idx = sentence.find(i)
        if idx != -1:
            sentence = sentence[:idx]
    sentence = sentence.lower()
    sentence = re.sub(r'http:\\*/\\*/.*?\s', ' ', sentence)# remove url
    sentence = re.sub(r'www.\S+', '', sentence)  # remove url
    sentence = re.sub(r'[\w\.-]+@[\w\.-]+', '', sentence)  # remove email
    sentence = re.sub(r'[\W_]+', ' ', sentence) # remove special characters
    sen = re.sub(r"[^a-zA-Z]", " ", sentence) # here exclude the number also
    sen =" ".join(lemmatizer.lemmatize(word) for word in sen.split())
    return sen


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def preprocessing():
    df1=pandas.read_csv(thefile)
    df1=df1[df1.content.str.len()>100]
    df1=df1[~df1.title.isin(['Featured Cannabis Companies','Alan Brochstein, CFA - Media Mentions'])]   #exclude some not useful data
    df1=df1.drop_duplicates(subset='title',keep='last')
    df1=df1[['content','title','date']]
    df1['content']=df1['content'].apply(lambda x: RemoveHTTP(str(x)))
    #print(df['content'].iloc[0])
    return df1


def tfidf_fit(df1):
    tvectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.6, min_df=5, max_features=1000)
    feature_vec = tvectorizer.fit_transform(df1['content']).toarray()
    return feature_vec,tvectorizer.get_feature_names()


def modeling_lda(x,num_t):
    lda=LDA(n_components=num_t,random_state=0)
    lda.fit(x)
    return lda


def print_close_docs(doc_em,corpus_emb,df1):
    """
    print the top 10 sentences close to specific doc
    """
    fs_dis=[cosine(corpus_emb[i],doc_em) for i in range(len(corpus_emb))]
    fs_0=np.argsort(np.array(fs_dis))
    for i in fs_0[:10]:
        print([df1.title.iloc[i]+"----"+df1.date.iloc[i]])
        print("=============================================")
    print([fs_dis[i] for i in fs_0[:10]])


def modeling_nmf():
    model_nmf=NMF(n_components=num_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(dtm)
    return model_nmf


def query_close_doc(idx):
    """
    :param idx: the idx in the corpus
    :return: the close documents
    """
    print_close_docs(dtm[idx], dtm,df)


def query_new_doc(content,title=None,date=None):
    """
    article retrieval with tf-idf vectorizer
    :param content: new article content
    :param title: title of new added articles
    :param date: the time when articles being published
    :return: the close documents
    """
    df.append({'content':content, 'title':title,'date':date}, ignore_index=True)
    fea_vec,fea_names=tfidf_fit(df)
    print_close_docs(fea_vec.iloc[-1],fea_vec,df)


if __name__ == '__main__':
    num_topics = 4
    df = preprocessing()
    dtm, feature_nmf = tfidf_fit(df)
    count_vectorizer = CountVectorizer(stop_words=stop_words)
    X_count = count_vectorizer.fit_transform(df['content'])
    feature_names = count_vectorizer.get_feature_names()
    lda_model = modeling_lda(X_count,num_topics)
    nmf_model = modeling_nmf()
    #df.tem_title = df.title.apply(lambda x: [x])
    #corpus = df.tem_title.agg(sum)
    #query_close_doc(101)
    # print(nmf_model.transform(dtm)[:10])
    #print(nmf_model.components_)
    #print(df1.content.iloc[100])
    #print()
    #print(len(dtm))
    #print(df.title.iloc[101])
    #print_close_docs(dtm[101],dtm,corpus)
    #print(lda_model.transform(X_count[1000]))
    #print(lda_model.transform(X_count[:3]))
    print_top_words(lda_model,feature_names,10)
    print_top_words(nmf_model,feature_nmf,10)
