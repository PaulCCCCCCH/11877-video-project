from PIL import Image
import os

class VideoFramesDataset(vis.datasets.VisionDataset):
  def __init__(self, root, transforms):
    self.categories = os.listdir(root)
    self.categories.sort()
    self.transcript_base = "/home/zhechen2/results/smit_selected/transcriptions"
    
    self.videos = list()
    for cat_idx, category in tqdm(enumerate(self.categories), ncols=100, total=len(self.categories), desc="dir walk"):
      for video in sorted(os.listdir(f"{root}/{category}")):
        self.videos.append((cat_idx, video, sorted(os.listdir(f"{root}/{category}/{video}"))))
    
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
    cat_idx, video, frames = self.videos[index]
    video_path_base = f"{self.categories[cat_idx]}/{video}"
    
    images = [self.transforms(self.pil_loader(f"{video_path_base}/{frame}")) for frame in frames]
    
    with open(f"{self.transcript_base}/{video_path_base[:-4]}.txt") as file:
      text = file.readline().strip()
    
    return torch.stack(images), len(images), text[:77] # clip context length defaults to 77

  def collate_fn(batch):
    frames, lengths, texts = list(zip(*batch))
    return torch.cat(frames), lengths, clip.tokenize(texts)