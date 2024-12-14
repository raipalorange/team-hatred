from ..data.datasets import TextDataset
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_combined_df(config):

    df_davidson = pd.read_parquet("hf://datasets/tdavidson/hate_speech_offensive/data/train-00000-of-00001.parquet")

    df_goldbeck = pd.read_csv(config['path']['goldbeck'], sep='\t', encoding_errors='ignore')
    df_devansh = pd.read_csv(config['path']['devansh'])
    df_reddit = pd.read_csv(config['path']['reddit'])

    ds = load_dataset("wisnu001binus/Hate_Speech_Dataset")
    df_wisnu = pd.concat([ds['train'].to_pandas(),ds['test'].to_pandas()],ignore_index=True)
    df_wisnu_hate = df_wisnu[df_wisnu['Label'] == 1]['Content']

    ds = load_dataset("badmatr11x/hate-offensive-speech")
    df_badmatr = pd.concat([ds['train'].to_pandas(),ds['test'].to_pandas(),ds['validation'].to_pandas()],ignore_index=True)
    df_badmatr_hate = df_badmatr[df_badmatr['label'].isin([0,1])]['tweet']

    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    df_ucb = ds['train'].to_pandas()
    df_ucb_hate = df_ucb[df_ucb['hate_speech_score']>=0]['text']

    df_reddit_hate = df_reddit[df_reddit['Hateful']==1]['Comment']

    df_goldbeck_hate = df_goldbeck[df_goldbeck['Code']=='H']['Tweet']

    df_davidson_hate = df_davidson[df_davidson['class'] != 2]['tweet']

    df_devansh_hate = df_devansh[df_devansh['Label'] ==1]['Content']

    df_combined_hate = pd.concat([df_davidson_hate,df_goldbeck_hate,df_devansh_hate,df_reddit_hate,df_ucb_hate,df_wisnu_hate,df_badmatr_hate],ignore_index=True)
    
    return df_combined_hate

def get_combined_dataset(config):

    texts = get_combined_df(config)
    dataset = TextDataset(texts)


    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    i = 0
    for batch in dataloader:
        print("Total",int(batch[-1]),"texts are done in iteration",i)
        i += 1

    return dataset.get_dataset()
