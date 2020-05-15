# Data Visualization
import matplotlib.pyplot as plt

# Data Processing
import pandas as pd

# NLP 
import re

# Others
import os
from collections import defaultdict

# extract Counsil Chat crawled data into a dataframe
def get_crawled_data_df(path):
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
    if 'Unnamed: 0' in df_crawled.columns:
        df_crawled.drop("Unnamed: 0", axis = 1, inplace = True)
    df_crawled = df_crawled[["questionTitle","questionText","questionLink","topic","answerText","upvotes","views"]]
    df_crawled["questionText"] = df_crawled["questionTitle"] + " " + df_crawled["questionText"]
    df_crawled = df_crawled.reset_index()
    df_crawled = df_crawled.drop(['index'], axis = 1)
    return df_crawled

def findLastIndex(strn): 
    li_strn = strn.split()
    index = -1
    for i in range(0, len(li_strn)): 
        if li_strn[i] == '?': 
            index = i 
    return index, li_strn, len(li_strn)

# process long questions because bert can only handle sentence with maximum of 512 tokens
def process_question_for_bert(question_df):
    
    for i in range(question_df.shape[0]):
        question_df.loc[i,"questionText"] = re.sub('\?\?+','?',question_df.loc[i,"questionText"])
        li = re.findall('(?<=\.|\?)[^\.]+?\?', question_df.loc[i,"questionText"])

        if li == []:
            li1 = re.findall('.*\?', question_df.loc[i,"questionText"])
            li1.sort(key = lambda x:len(x))
            ques = li1[-1]
        else:
            li.sort(key = lambda x:len(x))
            ques = li[-1]
        question_df.loc[i,"questionTitle"] = ques.strip()
        index, li_qt, context_len = findLastIndex(question_df.loc[i,"questionText"])
        pre_ques_len =  index 
        post_ques_len = context_len - index 
        if pre_ques_len > 300 and post_ques_len > 200:
            qt = ' '.join(li_qt[index-300:index] + li_qt[index:index+200])
        elif pre_ques_len > 300:
            if index-300-200+post_ques_len < 0:
                indx = 0
            else:
                indx = index-300-200+post_ques_len
            qt = ' '.join(li_qt[indx:index] + li_qt[index:index+post_ques_len])
        elif post_ques_len > 200:
            qt = ' '.join(li_qt[index-pre_ques_len:index] + li_qt[index:index+200+300-pre_ques_len])
        else:
            qt = ' '.join(li_qt)
        question_df.loc[i,"questionText"] = qt
    return question_df

# process data from kaggle
def get_kaggle_data_df(file):
    df_kaggle = pd.read_csv(file)
    df_kaggle.Answer = df_kaggle.Answer.astype(str)
    df_kaggle.Question = df_kaggle.Question.astype(str)
    if 'Unnamed: 0' in df_kaggle.columns:
        df_kaggle.drop(['Unnamed: 0'], inplace = True, axis = 1)
    question_df = df_kaggle[df_kaggle.Question.str.find("?") != -1]
    question_df = question_df.reset_index()
    question_df.drop(['index'], axis = 1, inplace = True)
    question_df.Question = question_df.Question.apply(string_processing)
    question_df.rename(columns={'Question': 'questionText','Answer':'answerText'}, inplace=True)
    question_df = process_question_for_bert(question_df)
    return question_df
    

# adds space before and after puntuations - preprocessing step
def string_processing(answer):
    pattern = '(?<! )(?=[!\"#$%&\'()\*\+,-\.:;=?@\[\]\\^_`\|\{\}~])|(?<=[!\"#$%&\'()\*\+,-\.:;=?@\[\]\\^_`\|\{\}~])(?! )'
    answer = re.sub(pattern, r' ',  answer)
    return answer

# generate topics and multi labels for crawled dataset to perform multiclass multilabel classification (used my baseline model)
def generate_root_topics_and_multi_labels(df_crawled):
    di = defaultdict(set)

    for i in range(df_crawled.shape[0]):
        if str(df_crawled.loc[i,"topic"]) in ["relationships","intimacy","family-conflict","parenting","relationship-dissolution","marriage","domestic-violence"]:
            df_crawled.loc[i,"root_topic"] = "family_conflicts"
        elif df_crawled.loc[i,"topic"] in ["depression","anxiety","stress","anger-management","trauma"]:
            df_crawled.loc[i,"root_topic"] = "emotional_conflicts"
        else:
            df_crawled.loc[i,"root_topic"] = "others"

        di[df_crawled.loc[i,"questionTitle"]].add(df_crawled.loc[i,"root_topic"])
    df_crawled.sort_values(by=['upvotes','views'], ascending=False, inplace = True)
    df_crawled.drop_duplicates(subset=['questionTitle','topic'], keep="first",inplace = True)
    df_crawled = df_crawled.reset_index()
    df_crawled = df_crawled.drop("index", axis = 1)
    labels = []
    for i in range(df_crawled.shape[0]):
        topics = list(di[df_crawled.loc[i,"questionTitle"]])
        label = [0,0,0]
        for topic in topics:
            if topic == 'family_conflicts':
                label[0] = 1
            if topic == 'emotional_conflicts':
                label[1] = 1
            if topic == 'others':
                label[2] = 1
        labels.append(label)
    df_crawled["root_multi_label"] = pd.Series(labels, dtype = object)
    return df_crawled

# generate reponses/reflections from therapist answers
def generate_responses(final_df):
    for i in range(final_df.shape[0]):
        # extract if a sentence in answer has sounds like or seems like
        if re.findall(r"([^.]*?(sounds like|seems like)[^.]*\.)",final_df.loc[i,"answerText"]):
            final_df.loc[i,"reflection"] = re.findall(r"([^.]*?(sounds like|seems like)[^.]*\.)",final_df.loc[i,"answerText"])[0][0] 
        # check if answer contains atleast 2 senetences, extract the second sentence if the first sentence is shorter (for example if "hello" is the first sentence) 
        # check oif hello or thank you is present in first sentence, then extract second sentence
        elif (
            (len(final_df.loc[i,"answerText"].split(".")[0].split()) <= 2 and len(final_df.loc[i,"answerText"].split(".")) >= 2)
            or (("hello" in final_df.loc[i,"answerText"].split(".")[0].lower() or "hi" in final_df.loc[i,"answerText"].split(".")[0].lower() or "thank you" in final_df.loc[i,"answerText"].split(".")[0].lower()) and len(final_df.loc[i,"answerText"].split(".")) >= 2)):
            final_df.loc[i,"reflection"] = final_df.loc[i,"answerText"].split(".")[1].strip()
        # else just extract the first sentence as reflection
        else:
            final_df.loc[i,"reflection"] = final_df.loc[i,"answerText"].split(".")[0].strip()
    return final_df

def main(): 
    # crawled data processing
    crawled_data_path = "../data/CouncilChat-CrawledData/"
    df_crawled = get_crawled_data_df(crawled_data_path)
    # kaggle data processing
    kaggle_dataset = '../data/Psych_data.csv'
    question_df = get_kaggle_data_df(kaggle_dataset)
    df_crawled = generate_root_topics_and_multi_labels(df_crawled)
    # concatenate dataframes
    final_df = pd.concat([df_crawled[["questionTitle","questionText","answerText"]], question_df[["questionTitle","questionText","answerText"]]])
    # processing concatenated dataset
    final_df = final_df.reset_index()
    final_df = final_df.drop("index", axis = 1)
    final_df.answerText = final_df.answerText.apply(string_processing)
    final_df = generate_responses(final_df)
    # generate reponses/reflections from therapist answers
    final_df = generate_responses(final_df)
    # pickle the dataset which is used in baseline and current models
    final_df.to_pickle("../data/preprocessed_dataset")

if __name__ == '__main__':
    main()

