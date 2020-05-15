# Data Processing
import pandas as pd

# BERT
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from scipy.spatial.distance import cosine

# loading pre-trained model tokenizer 
def initialize_tokenizer(pretrained_model):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    return tokenizer

# extracting sub token index 
def get_index(tokenized_concept, tokenized_text):
    result=[]
    sll=len(tokenized_concept)
    for ind in (i for i,e in enumerate(tokenized_text) if e==tokenized_concept[0]):
        if tokenized_text[ind:ind+sll]==tokenized_concept:
            result = [ind,ind+sll-1]
    return result

# returns contextual vector using pre-trained model
def get_contextual_vector(model, tokenizer, question, context):
    encoded_answer_context = "[CLS] " + question + " " + context + " [SEP]"
    # maximum words that BERT can have is 512
    encoded_answer_context = encoded_answer_context[:512]
    tokenized_encoded_answer_context = tokenizer.tokenize(encoded_answer_context)
    indexed_encoded_answer_context = tokenizer.convert_tokens_to_ids(tokenized_encoded_answer_context)
    segments_ids_encoded_answer_context= [1] * len(tokenized_encoded_answer_context)
    tokens_tensor_encoded_answer_context = torch.tensor([indexed_encoded_answer_context])
    segments_tensors_encoded_answer_context = torch.tensor([segments_ids_encoded_answer_context])
    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers_answer_context, _ = model(tokens_tensor_encoded_answer_context, segments_tensors_encoded_answer_context)
    token_vecs_answer_context= encoded_layers_answer_context[11][0]
    tokenized_answer = tokenizer.tokenize(question)
    indexes = get_index(tokenized_answer, tokenized_encoded_answer_context)
    if len(indexes) == 0:
        print ("check if question title is present in the question context")
        sys.exit()
    first_index, last_index = indexes[0], indexes[1] 
    return torch.mean(token_vecs_answer_context[first_index:last_index + 1], dim=0)