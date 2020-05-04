# Data Visualization
import matplotlib.pyplot as plt

# Data Processing
import pandas as pd

# NLP 
import re

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# BERT
from simpletransformers.classification import MultiLabelClassificationModel

import logging
import warnings


def main():
    df = pd.read_pickle("../data/preprocessed_dataset")

    df["text"] = df.questionText + ' ' + df.questionTitle
    df["text"] = df["text"].astype(str)
    df["labels"] = df.root_multi_label
    # defining model
    model = MultiLabelClassificationModel('bert', 'bert-base-uncased', num_labels=3, use_cuda=False)
    # processing train and test data for multilabel classification
    train_df, test_df = train_test_split(df, test_size=0.3, random_state =333)
    train_df = train_df[['text', 'labels']]
    test_df = test_df[['text', 'labels']]
    train_df = train_df.reset_index()
    train_df.drop(['index'], axis = 1, inplace = True)
    test_df = test_df.reset_index()
    test_df.drop(['index'], axis = 1, inplace = True)
    model.train_model(train_df, args={'learning_rate':1e-4, 'num_train_epochs': 10, 'reprocess_input_data': True, 'overwrite_output_dir': True,"train_batch_size": 14})
    # evaluate
    result, model_outputs, wrong_predictions = model.eval_model(test_df)
    
if __name__ == '__main__':
    main()


