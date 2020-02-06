from summarizer import Summarizer  # pretrained model to extract important sentences
import spacy  # with pretrained model to recognize entities
import pandas
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os
package_dir = os.path.dirname(os.path.abspath('__file__'))
thefile = os.path.join(package_dir, '../data/cleaned.csv')
nlp = spacy.load("en_core_web_lg")


def cloudW(df, file, column):
    text = df[column].str.cat(sep='. ')
    stopwords=set(STOPWORDS)
    stopwords.update(['cannabis', 'marijuana', 'will', 'company', 'release', 'press'])
    # fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    # wordcloud.to_file(file)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # plt.show()
    plt.savefig(file, dpi=300, bbox_inches='tight')
    plt.close()


def preprocessing(file):
    dataf = pandas.read_csv(file)
    dataf = dataf[['content', 'title', 'date']]
    dataf = dataf[df.content.str.len() > 100]
    dataf = dataf[~df.title.isin(['Featured Cannabis Companies','Alan Brochstein, CFA - Media Mentions'])]   # exclude some not useful data
    dataf = dataf.drop_duplicates(subset='title', keep='last')
    return dataf


if __name__ == "__main__":
    df = preprocessing(thefile)
    summarize = Summarizer()
    # the summary usually takes longer time, using parallel processing
    df['summary'] = df['content'].apply(lambda x: summarize(x, min_length=100))
    df['ents'] = df['summary'].apply(lambda x: nlp(str(x)).ents)
    df['ORG'] = df.ents.apply(lambda x: [i.text for i in x if i.label_ == 'ORG'])
    orgs = ['Tweed', 'Aurora', 'GW', 'Canopy Growth', 'Aphria', 'FDA', 'Health Canada', 'Shoppers Drug Mart']
    for org in orgs:
        df[org] = df['ORG'].apply(lambda x: 1 if org in '. '.join(i for i in x) else 0)
    df['GPE'] = df.ents.apply(lambda x: [i.text for i in x if i.label_ == 'GPE'])
    df['Ca'] = df['GPE'].apply(
        lambda x: 1 if 'Canada' in x or 'Ontario' in x or 'Alberta' in x or 'British Columbia' in x else 0)
    df.to_csv(os.path.join(package_dir, 'summary_ents'))