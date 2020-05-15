# Data Processing
import pandas as pd

# BERT
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from scipy.spatial.distance import cosine
from utils import initialize_tokenizer, get_contextual_vector

def main():

    df = pd.read_pickle("../data/preprocessed_dataset")
    # defining pretrained model
    pretrained_model = 'bert-base-uncased'
    tokenizer = initialize_tokenizer(pretrained_model)
    model = BertModel.from_pretrained(pretrained_model)

    # starting feedforward network
    model.eval()

    # computing contextual vectors for all entries in whole dataset
    contextual_vectors = []
    for i in range(df.shape[0]):
        try:
            contextual_vectors.append(get_contextual_vector(model, tokenizer, df.loc[i,"questionTitle"], df.loc[i,"questionText"]))
        except:
            contextual_vectors.append([0 for i in range(768)])

    df["contextual_vector"] = pd.Series(contextual_vectors)

    # saving the processed dataframe
    df = pd.read_pickle("../data/contextual_similarity_df")

if __name__ == '__main__':
    main()