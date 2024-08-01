export CUDA_VISIBLE_DEVICES=0

python -c "import torch; torch.cuda.empty_cache()"

export CUDA_LAUNCH_BLOCKING=1
# Draem
python main.py --cls bean --model draem --root "../../agricultural_dataset" --batch_size 32 --img_size 128 

# PatchCore
python main.py --cls bean --model patchcore --root "../../agricultural_dataset" --batch_size 16 --img_size 128

export CUDA_VISIBLE_DEVICES=1

python -c "import torch; torch.cuda.empty_cache()"

export CUDA_LAUNCH_BLOCKING=1

# EfficientAD
python main.py --cls bean --model efficientad --root "../../agricultural_dataset" --batch_size 1 --img_size 128
