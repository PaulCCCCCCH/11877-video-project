set -ex
python train.py --dataroot dataset/bear/ --checkpoints_dir checkpoints/ --name bear_256x512 --batch_size 12 --display_id 0 --model visa --dataset_mode visa --niter 100 --niter_decay 100 --norm batch --gpu 0 --ngf 64 --print_freq 100 --loadSize_h 256 --loadSize_w 512
