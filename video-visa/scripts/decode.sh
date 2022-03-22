set -ex
python test.py --dataroot results/bear/latent_code/bear_256x512/test_latest/images/ --checkpoints_dir checkpoints/ --name bear_256x512 --num_test 25000000 --model visa  --dataset_mode visa --norm batch --decode_only True --gpu 0 --ngf 64 --loadSize_h 256 --loadSize_w 512  --results_dir results/bear/decode/ --eval

