import torch
from torch.utils.data import Dataset, DataLoader
import torchtext
import MeCab
from torchtext.data import get_tokenizer

# Stored Tokens 
EnglishTokens = []
KoreanTokens = []

# Tokenizing, splitting "Hello, Cheese" into ['Hello','Cheese'] and "유양 씨는" into ['유양', '씨는'] subject to markers like '이'
EnglishTokenizer = get_tokenizer("basic_english")
KoreanTokenizer = MeCab.Tagger()

# Get Dataset, and split into Korean and English then Tokenize & store tokens
with open('./Dataset/ParallelData.txt') as Dataset:
    for LineSentence in Dataset:
        Sentence = LineSentence.split('/')
        EnglishTokens.append(EnglishTokenizer(Sentence[1]))
        KoreanTokens.append(KoreanTokenizer.parse(Sentence[0]).split())

# The model input is total vocabulary size of English since we are translating English to Korean, the output is the vocab size of Korean
ModelInput = len(EnglishTokens)
ModelOutput = len(KoreanTokens)

# Dataset Loader for feeding Tokens into Model
class TokenLoaderDataset(Dataset):

    # Initialize
    def __init__(self, ModelInput, ModelOutput):
        self.ModelInput = ModelInput
        self.ModelOutput = ModelOutput

    # Len is a Dataloader function for returning the total amount of items in the dataset
    def __len__(self):
        return len(self.ModelInput)
    
    # Get Item calls every time the model requires new data, giving index from the total of the data
    def __getitem__(self, index):
        return self.ModelInput[index], self.ModelOutput[index]

# Initialize and Create Dataloader
TokenData = TokenLoaderDataset(ModelInput,ModelOutput)
batch_size = 32
TokenLoader = DataLoader(TokenLoaderDataset, batch_size=batch_size, shuffle=True)
