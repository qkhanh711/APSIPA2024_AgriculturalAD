# EfficientAD
python main.py --cls bean --model efficientad --root "../../../patchcore-inspection/mvtec_anomaly_detection" --batch_size 1

# Draem
python main.py --cls bean --model draem --root "../../../patchcore-inspection/mvtec_anomaly_detection" --batch_size 32 --img_size 128

# PatchCore
python main.py --cls bean --model patchcore --root "../../../patchcore-inspection/mvtec_anomaly_detection" --batch_size 16 --img_size 128