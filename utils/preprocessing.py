
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
import string
import torch                             
from gensim.models import Word2Vec  
from models.hatred_prime import hatred_prime_train        



def convert_sequences_to_tensor(sequences,config):

    num_sequences = len(sequences)
    if config['model']['trained']:
        model = Word2Vec.load(config['path']['word2vec_model_path'])
    else:
       model = hatred_prime_train()


    data_tensor = torch.zeros((num_sequences, config['training']['sequence_length'],config['model']['input_size']))

    for index, review in enumerate(list(sequences)):

        truncated_clean_review = review[:config['training']['sequence_length']] # truncate to sequence length limit
        list_of_word_embeddings = [model.wv[word] if word in model.wv else [0.0]*config['model']['input_size'] for word in truncated_clean_review]

        sequence_tensor = torch.FloatTensor(list_of_word_embeddings)

        # add the review to our tensor of data
        review_length = sequence_tensor.shape[0] # (review_length, embedding_size)
        data_tensor[index,:review_length,:] = sequence_tensor

    return data_tensor


def preprocess(text):

  stop_words = set(stopwords.words('english'))
  text = re.sub(r'@\w+', '', text)
  text = re.sub(r'RT\s', '', text)

  text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation


  words = nltk.word_tokenize(text.lower()) # lowercase
  
  words = [word for word in words if word not in stop_words]


  return words

def clean_davidson_tweets():
    
    df_davidson = pd.read_parquet("hf://datasets/tdavidson/hate_speech_offensive/data/train-00000-of-00001.parquet")

    df_davidson['class'].replace({0: 0, 1: 0, 2: 1}, inplace=True)
    df = df_davidson[['class','tweet']]
    df_train, df_test = train_test_split(df, test_size=0.2)

    df_train['clean_tweet'] = df_train['tweet'].apply(lambda str_: preprocess(str_))
    df_test['clean_tweet'] = df_test['tweet'].apply(lambda str_: preprocess(str_))

    return df_train,df_test