set -ex
python test.py --dataroot ../pre-processing/subset_classes --checkpoints_dir checkpoints/ --name bear_256x512 --num_test 25000000 --model visa  --dataset_mode visa --norm batch --latent_code_only True --gpu 0 --ngf 64 --loadSize_h 256 --loadSize_w 512  --results_dir results/bear/latent_code/ --eval