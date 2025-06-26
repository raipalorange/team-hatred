from torch.utils.data import DataLoader, Dataset
from nltk.corpus import stopwords
import nltk
import string

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        self.stop_words = set(stopwords.words('english'))
        self.dataset = [["hello","bitch"]]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx,alarm=False):
        if alarm:
            return len(self.dataset)

        text = self.texts[idx]

        # Clean the text
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation

        words = nltk.word_tokenize(text.lower()) # lowercase
        words = [word for word in words if word not in self.stop_words]

        self.dataset.append(words)

        return len(self.dataset)

    def get_dataset(self):

        return self.dataset
