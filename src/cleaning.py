import pandas,os
from flatten_json import flatten
import sys
sys.path.append("./News_articles")
os.chdir('../News_articles')
def data_clean():
    lines=pandas.read_json('merged1.json',lines=True)
    lines=lines.posts
    #df=pandas.DataFrame(lines[0])
    result=[]
    result.append(lines[0][0])
    for i in lines[1:]:
        #print(i)
        result.append(i[0])
        #df.append(i[0],ignore_index=True,sort=False)
    #print(type(result[0][0]))
    print(len(result))
    #for i in result:
    #    print(i) 
    #    print(type(i[0]))
    flatten_dic=[flatten(d) for d in result]
    df = pandas.DataFrame(flatten_dic)
    df.to_csv('samples.csv')
    cleaned=df.drop_duplicates(subset='content',keep='first')
    cleaned=cleaned[['title','content','date']]
    cleaned.to_csv('cleaned.csv')
