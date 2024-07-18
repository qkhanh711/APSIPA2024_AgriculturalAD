export CUDA_VISIBLE_DEVICES=0

python -c "import torch; torch.cuda.empty_cache()"

export CUDA_LAUNCH_BLOCKING=1
 # Draem
 python main.py --cls bean --model draem --root "../../../patchcore-inspection/mvtec_anomaly_detection" --batch_size 32 --img_size 128

 export CUDA_VISIBLE_DEVICES=1

 python -c "import torch; torch.cuda.empty_cache()"

 export CUDA_LAUNCH_BLOCKING=1

# PatchCore
python main.py --cls bean --model patchcore --root "../../../patchcore-inspection/mvtec_anomaly_detection" --batch_size 16 --img_size 128

# EfficientAD
 python main.py --cls bean --model efficientad --root "../../../patchcore-inspection/mvtec_anomaly_detection" --batch_size 1 --img_size 128
