# BizPulse: Connect people to content

## Overview
Data science project at Insight Data Science 2020 January. This is a project with Right Strain to link users with similar topic news articles. 

## Problem
The data source is news articles from [new cannabis ventures](https://www.newcannabisventures.com) from 2015 August to 2019 December. First approach for this problem, topic modeling is used on all news articles to extract topics. However, topics with topic modeling pops out companies and locations rather than different topics like investment, products or new policies.

## Solution
Since many of the articles contain too many entities like companies and locations, first step is to extract the import entities. Without the labels of entities, spaCy with pre-trained Name Entity Recognition models is applied to identity the entities (mainly companies, organizations and locations) within content of the articles.

To link users to similar stories, titles and key sentecnes in news are embedding with sentence transformer. With cosine similarity, users can retrieve articles with the similar topics to the input title from user. This article recommendation web-app is based on the semantic searching of titles and key sentences within content.
![Alt text](schema.png?raw=true "Title")

The pipeline is first cleaning the data, aggregating all news and reducing the duplicates terms, url, emails etcs. To retrieve the similar topic stories with specific company, semantic search based on cosine similarity of titles and key sentences with sentence transformer is employed.

src contains all the source code include data clean, data preprocessing, exploratory data analysis, topic modeling, article summarizer with transformer and NER, sentence embedding etc.
## Demo
Web-app is built with flask on aws. Try on the website [Sites Using flask](http://www.dsprojectsz.club:5000).

## Prerequisite

## Reference
*[Sentence BERT](https://github.com/UKPLab/sentence-transformers): Fine tuning BERT on semantic similarity 
*[ROUGE-n](http://nlpprogress.com/english/summarization.html): Metrics to evaluate the performance of articles' summary
* [Extractive summary bert](https://github.com/dmmiller612/bert-extractive-summarizer)
