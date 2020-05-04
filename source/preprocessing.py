# Data Visualization
import matplotlib.pyplot as plt

# Data Processing
import pandas as pd

# NLP 
import re

# Others
import os
from collections import defaultdict

def main():
    df_given = pd.read_csv('../data/given_counsel_chat_dataset.csv')
    path = "../data/CouncilChat-CrawledData/"
    files = os.listdir(path)
    files_xls = [f for f in files if f[-4:] == 'xlsx']
    list_df = []
    temp = []
    for index, filename in enumerate(files_xls):
        try: 
            list_df.append(pd.read_excel(path+filename))
            temp.append(filename.split(".")[0])
        except:
            temp.append()
            
    # combining all the crawled data
    df_crawled = pd.concat(list_df)
    df_crawled.dropna(inplace = True)
    df_crawled.drop("Unnamed: 0", axis = 1, inplace = True)
    # concatenating two datasets
    df = pd.concat([df_given[["questionTitle","questionText","questionLink","topic","answerText","upvotes","views"]],df_crawled[["questionTitle","questionText","questionLink","topic","answerText","upvotes","views"]]])
    df = df.reset_index()
    df = df.drop("index", axis = 1)
    di = defaultdict(set)
    # creating root topics
    for i in range(df.shape[0]):
        if str(df.loc[i,"topic"]) in ["relationships","intimacy","family-conflict","parenting","relationship-dissolution","marriage","domestic-violence"]:
            df.loc[i,"root_topic"] = "family_conflicts"
        elif df.loc[i,"topic"] in ["depression","anxiety","stress","anger-management","trauma"]:
            df.loc[i,"root_topic"] = "emotional_conflicts"
        else:
            df.loc[i,"root_topic"] = "others"

        di[df.loc[i,"questionTitle"]].add(df.loc[i,"root_topic"])
    df.sort_values(by=['upvotes','views'], ascending=False, inplace = True)
    df.drop_duplicates(subset=['questionTitle','topic'], keep="first",inplace = True)
    df = df.reset_index()
    df = df.drop("index", axis = 1)
    labels = []
    for i in range(df.shape[0]):
        topics = list(di[df.loc[i,"questionTitle"]])
        label = [0,0,0]
        for topic in topics:
            if topic == 'family_conflicts':
                label[0] = 1
            if topic == 'emotional_conflicts':
                label[1] = 1
            if topic == 'others':
                label[2] = 1
        labels.append(label)

    df["root_multi_label"] = pd.Series(labels, dtype = object)
    # extracting reflections
    for i in range(df.shape[0]):
    # extract if a sentence in answer has sounds like or seems like
    if re.findall(r"([^.]*?(sounds like|seems like)[^.]*\.)",df.loc[i,"answerText"]):
        df.loc[i,"reflection"] = re.findall(r"([^.]*?(sounds like|seems like)[^.]*\.)",df.loc[i,"answerText"])[0][0] 
    # check if answer contains atleast 2 senetences, extract the second sentence if the first sentence is shorter (for example if "hello" is the first sentence) 
    # check oif hello or thank you is present in first sentence, then extract second sentence
    elif ((len(df.loc[i,"answerText"].split(".")[0].split()) <= 2 and len(df.loc[i,"answerText"].split(".")) >= 2) or (("hi" in df.loc[i,"answerText"].split(".")[0].lower() or "hello" in df.loc[i,"answerText"].split(".")[0].lower() or "thank you" in df.loc[i,"answerText"].split(".")[0].lower()) and len(df.loc[i,"answerText"].split(".")) >= 2)):
        df.loc[i,"reflection"] = df.loc[i,"answerText"].split(".")[1].strip()
    # else just extract the first sentence as reflection
    else:
        df.loc[i,"reflection"] = df.loc[i,"answerText"].split(".")[0].strip()
    df.to_pickle("../data/preprocessed_dataset")
    
if __name__ == '__main__':
    main()

