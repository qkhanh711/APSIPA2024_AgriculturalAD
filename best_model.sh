rm -r ~/Code/Research/computer_vision/patchcore-inspection/mvtec_anomaly_detection/bean
# Move obects selected by IoU to the target folder
cd ~/Code/Research/computer_vision/EvaluateAD/APSIPA2024_AgriculturalAD/trainingOD
python move.py --f 0.75

# Training EfficientAD and creating anomaly maps
cd ~/Code/Research/computer_vision/EfficientAD-modify
bash run.sh

# Run AD models in Anomalib
cd ~/Code/Research/computer_vision/EvaluateAD/APSIPA2024_AgriculturalAD/trainingAD/
export CUDA_VISIBLE_DEVICES=0

python -c "import torch; torch.cuda.empty_cache()"

export CUDA_LAUNCH_BLOCKING=1
# Draem
python main.py --cls bean --model draem --root "../../../patchcore-inspection/mvtec_anomaly_detection" --batch_size 32 --img_size 128 --f 0.75


rm -r ~/Code/Research/computer_vision/patchcore-inspection/mvtec_anomaly_detection/bean
# Move obects selected by IoU to the target folder
cd ~/Code/Research/computer_vision/EvaluateAD/APSIPA2024_AgriculturalAD/trainingOD
python move.py --f 0.9

# Training EfficientAD and creating anomaly maps
cd ~/Code/Research/computer_vision/EfficientAD-modify
bash run.sh

# Run AD models in Anomalib
cd ~/Code/Research/computer_vision/EvaluateAD/APSIPA2024_AgriculturalAD/trainingAD/
export CUDA_VISIBLE_DEVICES=0

python -c "import torch; torch.cuda.empty_cache()"

export CUDA_LAUNCH_BLOCKING=1

# PatchCore
python main.py --cls bean --model patchcore --root "../../../patchcore-inspection/mvtec_anomaly_detection" --batch_size 16 --img_size 128 --f 0.9



export CUDA_VISIBLE_DEVICES=1

python -c "import torch; torch.cuda.empty_cache()"

export CUDA_LAUNCH_BLOCKING=1

# EfficientAD
python main.py --cls bean --model efficientad --root "../../../patchcore-inspection/mvtec_anomaly_detection" --batch_size 1 --img_size 128 --f 0.9

