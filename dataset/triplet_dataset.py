import numpy as np
import torch
from torch.utils import data
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import pandas as pd

_tokenizer = _Tokenizer()

# Modified based on CLIP's tokenize function
from typing import Union, List
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = True):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    lengths = list()

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
        lengths.append(len(tokens))

    return result, torch.tensor(lengths, dtype=torch.long)

class TextTripletDataset(data.Dataset):
  def __init__(self, split, transcript_base, num_repeats=1):
    split = pd.read_csv(split)
    self.cat_to_texts = list()
    self.categories = dict()
    for _, (cat, vid) in split.iterrows():
      if cat in self.categories:
        idx = self.categories[cat]
      else:
        idx = len(self.categories)
        self.categories[cat] = idx
        self.cat_to_texts.append(list())

      with open(f"{transcript_base}/{cat}/{vid}.txt") as file:
        text = file.readline().strip()
        self.cat_to_texts[idx].append(text)
    self.num_repeats = num_repeats

  def __len__(self):
    return len(self.categories) * self.num_repeats
    
  def __getitem__(self, idx):
    anchor_class_idx  = idx % len(self.categories)
    """Treat the given index as the anchor class and pick a triplet randomly"""
    anchor_class = self.cat_to_texts[anchor_class_idx]
    # choose positive pair (assuming each class has at least 2 images)
    anchor, positive = np.random.choice(a=anchor_class, size=2, replace=False)
    # choose negative image
    # hint: you can choose 2 negative images to make it a Quadruplet Loss
    all_classes = list(range(len(self.categories)))
    all_classes.pop(anchor_class_idx)
    negative_class = np.random.choice(a=all_classes, size=1, replace=False)[0]
    negative = np.random.choice(a=self.cat_to_texts[negative_class], size=1, replace=False)[0]
    
    # [0] removes class label from the built-in dataset
    return anchor, positive, negative

  @staticmethod
  def collate_fn(batch):
    anchors, positives, negatives = list(zip(*batch))
    return tokenize(anchors), tokenize(positives), tokenize(negatives)