#!/bin/bash

export BASE_DIR=smit_selected/videos

for video_name in $(ls $BASE_DIR); do
  mkdir -p extracted_audio/$video_name
  mkdir -p extracted_frames/$video_name
  for video_segment in $(cd $BASE_DIR/$video_name && ls *.mp4); do
    mkdir -p extracted_frames/$video_name/$video_segment
    ffmpeg -i $BASE_DIR/$video_name/$video_segment -vf fps=2/1 extracted_frames/$video_name/$video_segment/%03d.jpg
    ffmpeg -i $BASE_DIR/$video_name/$video_segment -vn -acodec copy extracted_audio/$video_name/$video_segment.aac;
  done;
done;
