import numpy as np
import os
import sys
import ntpath
import time
from . import util
from . import html
from scipy.misc import imresize
import scipy.io as sio

def save_associated_frames(image_dir, visuals, image_path, frame_no, aspect_ratio=1.0, width=256, input_name=None):
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        new_image_dir = os.path.join(image_dir, frame_no)
        if not os.path.exists(new_image_dir):
            os.mkdir(new_image_dir)
        save_path = os.path.join(new_image_dir, image_name) if input_name is None else os.path.join(new_image_dir, input_name + "_" + image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        util.save_image(im, save_path)

def save_associated_frames_similarities(image_dir, frames, frame_no):
    new_image_dir = os.path.join(image_dir, frame_no)
    if not os.path.exists(new_image_dir):
        os.mkdir(new_image_dir)
    association_file = os.path.join(new_image_dir, "association_cosine_similarities.txt")
    with open(association_file, 'w') as f:
        f.write("Image, Cosine Similarity\n")
        for sim in frames:
            f.write(str(sim[2]).zfill(6) + ", " + str(sim[1]) + "\n")

def save_frames(image_dir, visuals, image_path, aspect_ratio=1.0, width=256):
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
        util.save_image(im, save_path)

def save_real_frames(image_dir, visuals, image_path, aspect_ratio=1.0, width=256):
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    for label, im_data in visuals.items():
        if label != "reconst":
            im = util.tensor2im(im_data)
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            if aspect_ratio < 1.0:
                im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            util.save_image(im, save_path)