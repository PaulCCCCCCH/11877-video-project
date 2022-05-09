from PIL import Image
import os
import torchvision as vis
import torch
from .triplet_dataset import tokenize

class VideoFramesDataset(vis.datasets.VisionDataset):
  def __init__(self, root, transcript_base, transforms):
    self.categories = os.listdir(root)
    self.categories.sort()
    self.transcript_base = transcript_base
    
    self.videos = list()
    for category in tqdm(self.categories, ncols=100, total=len(self.categories), desc="dir walk"):
      for video in sorted(os.listdir(f"{root}/{category}")):
        frames = os.listdir(f"{root}/{category}/{video}")
        if len(frames) < 6:
          continue
        frames = frames[:6]
        frames.sort()
        self.videos.append((category, video, frames))
    
    self.root = root
    self.length = len(self.videos)
    self.transforms = transforms
        
  def __len__(self):
      return self.length
    
  def pil_loader(self, path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(f"{self.root}/{path}", 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

  def __getitem__(self, index):
    # only return frame-wise features for now
    category, video, frames = self.videos[index]
    video_path_base = f"{category}/{video}"
    
    images = [self.transforms(self.pil_loader(f"{video_path_base}/{frame}")) for frame in frames]
    
    with open(f"{self.transcript_base}/{video_path_base[:-4]}.txt") as file:
      text = file.readline().strip()
    
    return torch.stack(images, dim=1), len(images), text

  def collate_fn(batch):
    frames, lengths, texts = list(zip(*batch))
    return torch.stack(frames), lengths, tokenize(texts)
  
import pandas as pd

class VideoFramesDatasetWithSplit(vis.datasets.VisionDataset):
  def __init__(self, split, root, transcript_base, transforms):
    self.transcript_base = transcript_base
    
    split = pd.read_csv(split)
    self.videos = [(cat, vid, [f"{f:03d}.jpg" for f in range(1, 7)]) for _, (cat, vid) in split.iterrows()]
    
    self.root = root
    self.length = len(self.videos)
    self.transforms = transforms
        
  def __len__(self):
      return self.length
    
  def pil_loader(self, path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(f"{self.root}/{path}", 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

  def __getitem__(self, index):
    # only return frame-wise features for now
    category, video, frames = self.videos[index]
    video_path_base = f"{category}/{video}"
    
    images = [self.transforms(self.pil_loader(f"{video_path_base}.mp4/{frame}")) for frame in frames]
    
    with open(f"{self.transcript_base}/{video_path_base}.txt") as file:
      text = file.readline().strip()
    
    return torch.stack(images, dim=1), len(images), text

  def collate_fn(batch):
    frames, lengths, texts = list(zip(*batch))
    return torch.stack(frames), lengths, tokenize(texts)