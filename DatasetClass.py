# Required Libraries
import torch
from torch.utils.data import Dataset
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# NTLK Download if Punkt is not already installed for tokenizing
nltk.download('punkt')

# Create Dataset class
class EKMLDataset(Dataset):
    
    # Class Initialization
    def __init__(self, Datasetfile, transform):
        self.LanguageData = pd.read_csv(Datasetfile, header=None)
        self.transform = transform
    
    # Return the total number of lines of data
    def __len__(self):
        return len(self.data)
    
    # Return the data of the line specified (idx)
    def __getitem__(self, idx):
        
        # Set the Korean Translation
        Korean_Hangul = self.data.iloc[idx, 1]
        English_Text = self.data.iloc[idx, 2]

        # Tokenize the text to feed into the model
        Korean_Hangul, English_Text = self.transform(Korean_Hangul, English_Text)

        # Feed data into the model
        return Korean_Hangul, English_Text
    
# Tokenizer Transform
def tokenizer(Korean_Hangul, English_Text):
    
    # Tokenize Text
    Tokenized_Korean = word_tokenize(Korean_Hangul)
    Tokenized_English = word_tokenize(English_Text)

    # Return Tokenized Data
    return Tokenized_Korean, Tokenized_English