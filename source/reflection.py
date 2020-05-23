# Data Processing
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# BERT
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from scipy.spatial.distance import cosine
from utils import initialize_tokenizer,get_index, get_contextual_vector

def main():
    df = pd.read_pickle("../data/contextual_similarity_df")
    print ("Initiating CURTIS!...") 
    pretrained_model = 'bert-base-uncased'
    tokenizer = initialize_tokenizer(pretrained_model)
    model = BertModel.from_pretrained(pretrained_model)
    # starting feed forward network
    model.eval()
    print ("CURTIS: Hey Kaushik, I am happy to have you here again!")
    _ = input("CURTIS: How are you?\nKaushik: ")
    user_question = input("CURTIS: You are having a rough time \nCURTIS: What makes you feel like this?\nKaushik: ")
    user_question_context = input("CURTIS: Please provide more context for your problem\nKaushik: ")
    user_vec = get_contextual_vector(model, tokenizer, user_question, user_question_context)
    for i in range(df.shape[0]):
        try:
            cs_dist = cosine(user_vec, df.loc[i, "contextual_vector"])
        except:
            cs_dist = 1
        df.loc[i, "similarity_score"] = 1 - cs_dist
    print ("CURTIS:", df[df.similarity_score == df.similarity_score.max()].reset_index().reflection[0])
    
if __name__ == '__main__':
    main()
