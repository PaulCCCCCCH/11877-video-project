# For a video, converts video frames into video embedding (stacked image embeddings)
def encode_video(video_frames):
	batch_size = 256 # Process in batches for speed
	batches = math.ceil(len(video_frames) / batch_size)
	video_features = torch.empty([0, 1024], dtype=torch.float16).to(device)
	# In batches, convert video into stacked image embeddings
	for i in range(batches):
		batch_frames = video_frames[i * batch_size:(i + 1) * batch_size]
		batch_preprocessed = torch.stack(
			[preprocess(frame) for frame in batch_frames]).to(device)
		with torch.no_grad():
			batch_features = language_and_vision_model.encode_image(batch_preprocessed)
			batch_features /= batch_features.norm(dim=-1, keepdim=True)
		video_features = torch.cat((video_features, batch_features))
	return video_features