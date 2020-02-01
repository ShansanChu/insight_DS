from summarizer import Summarizer #pretrained model to extract important sentences
import spacy # with pretrained model to recognize entities
import pandas
nlp = spacy.load("en_core_web_lg")
def preprocessing(file):
    df=pandas.read_csv(file)
    df=df[['content','title','date']]
    df=df[df.content.str.len()>100]
    df=df[~df.title.isin(['Featured Cannabis Companies','Alan Brochstein, CFA - Media Mentions'])]   #exclude some not useful data
    df=df.drop_duplicates(subset='title',keep='last')
    return df
if __name__=="__main__":
    file='cleaned.csv'
    df=preprocessing(file)
    summarize=Summarizer()
    df['summary']=df['content'].apply(lambda x: summarize(x,min_length=100))
    #df['ents']=df['summary'].apply(lambda x: nlp(str(x)).ents)
    df.to_csv(summary_csv)