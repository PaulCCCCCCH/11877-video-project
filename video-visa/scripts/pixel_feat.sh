set -ex
python test.py --dataroot dataset/bear/ --checkpoints_dir checkpoints/ --name bear_256x512 --num_test 25000000 --model visa  --dataset_mode visa --norm batch --pixel_feat_only True --gpu 0 --ngf 64 --loadSize_h 256 --loadSize_w 512  --results_dir results/bear/pixel_feat/ --eval
