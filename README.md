# insight_DS
Data science project at Insight Data Science 2020 January. This is consulting project with Right Strain to recommend similar topic news articles. First approach for this problem, I used bag of words as well as tf-idf to perform topic modeling. However, the topics with topic modeling pops out companies and locations rather than different topics like investment, products or new policies.

Since many of the articles contain companies and locations, first step is to extract the import entities. Without the labels of entities, spaCy with pre-trained Name Entity Recognition models is applied to identity the entities (mainly companies, organizations and locations) within content of the articles.

To link users to similar stories, titles of news are embedding with sentence transformer. Based on cosine similarity of the titles, users can retrieve articles of the similar topics with a specific company. This article recommendation web-app is based on the semantic searching of titles and key sentences within content.
![Alt text](schema.png?raw=true "Title")

The pipeline is first cleaning the data, aggregating all news and reducing the duplicates terms, url, emails etcs. To retrieve the similar topic stories with specific company, semantic search based on cosine similarity of titles and key sentences with sentence transformer is employed.

src contains all the source code include data clean, data preprocessing, exploratory data analysis, topic modeling, article summarizer with transformer and NER, sentence embedding etc.
