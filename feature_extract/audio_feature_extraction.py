import numpy as np
import torch
import librosa
from PIL import Image
import wav2clip
import os
import os.path as osp

device = "cuda" if torch.cuda.is_available() else "cpu"
audio_model = wav2clip.get_model()

BASE_DIR = './extracted_audio'
FEATURE_DIR = './features_audio'
if not os.path.exists(FEATURE_DIR):
    os.mkdir(FEATURE_DIR)

audio_dict = {}

def extract_audio_features(src: str, dst: str):
    audio, sr = librosa.load(src)
    audio_features = wav2clip.embed_audio(audio, audio_model)
    np.save(dst, audio_features)
    return audio_features


for class_name in os.listdir(BASE_DIR):
    print("Extracting class {}".format(class_name))
    base_class_dir = osp.join(BASE_DIR, class_name)
    if not os.path.isdir(base_class_dir):
        continue

    feature_class_dir = osp.join(FEATURE_DIR, class_name)
    if not os.path.exists(feature_class_dir):
        os.mkdir(feature_class_dir)

    for video_name in os.listdir(base_class_dir):
        base_audio_path = osp.join(base_class_dir, video_name)
        feature_audio_path = osp.join(feature_class_dir, video_name)     
        features = extract_audio_features(base_audio_path, feature_audio_path)
        audio_dict["{}/{}".format(class_name, video_name)] = features

np.save('feature_dict.npy', features)
