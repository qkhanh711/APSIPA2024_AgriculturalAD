export CUDA_VISIBLE_DEVICES=1

python -c "import torch; torch.cuda.empty_cache()"

export CUDA_LAUNCH_BLOCKING=1
# python run_with_submitit.py --timeout 3000 --coco_path ../data_for_detr/
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path ../data_for_detr/