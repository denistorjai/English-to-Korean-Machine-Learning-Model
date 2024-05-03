# Required Libraries
import torch
from torch.utils.data import DataLoader

# Import Dataset
import DatasetClass
import LearningModel

# Training Data 
Training_Dataset = DatasetClass.EKMLDataset('./Dataset/KoreantoEnglishParallelData.csv', transform=DatasetClass.tokenizer)
TrainingDataLoader = DataLoader(Training_Dataset, batch_size=64, shuffle=False)

# Testing Data
Testing_Dataset = DatasetClass.EKMLDataset('./Dataset/KoreantoEnglishParallelTestingData.csv', transform=DatasetClass.tokenizer)
TestingDataLoader = DataLoader(Testing_Dataset, batch_size=64, shuffle=False)


