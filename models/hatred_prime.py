       
from gensim.models import Word2Vec   
from ..utils.data_loader_embeddings import get_combined_dataset

def hatred_prime_train(config):

    dataset = get_combined_dataset(config)
    
    model = Word2Vec(sentences=dataset, window=5, min_count=5, workers=4,vector_size=150)
    model.save(config['path']['word2vec_model_path'])

    return model