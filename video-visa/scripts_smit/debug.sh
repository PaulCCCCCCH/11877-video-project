python train.py --dataroot ../pre-processing/mmdata --checkpoints_dir checkpoints/ --name debug --batch_size 32 --display_id 0 --model visamm --dataset_mode visamm --niter 100 --niter_decay 100 --norm batch --gpu 0,1,2,3 --ngf 64 --print_freq 100 --loadSize_h 256 --loadSize_w 256