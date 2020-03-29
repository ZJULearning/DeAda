project_name='resnet50_training_method_plain_market_epoch50_stepsize10_lr_0.1'
mkdir logs
mkdir "logs/$project_name"
CUDA_VISIBLE_DEVICES=1 python main.py \
--project_name $project_name \
--dataset 'market1501-list' \
--b 48 \
--lr 0.1 \
--a resnet50 \
--training_method plain \
--data_dir '/home/longwei.wl/reid/open_reid_weilong/datasets' \
--features 512 \
--ncls 755 \
--step_size 10 \
--epochs 30 
