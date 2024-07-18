# Move obects selected by IoU to the target folder
cd ~/Code/Research/computer_vision/EvaluateAD/APSIPA2024_AgriculturalAD/crop_data
python move.py --f 0.5

# Training EfficientAD and creating anomaly maps
cd ~/Code/Research/computer_vision/EfficientAD-modify
bash run.sh

# Run AD models in Anomalib
cd ~/Code/Research/computer_vision/EvaluateAD/APSIPA2024_AgriculturalAD/
bash run.sh

# Move obects selected by IoU to the target folder
cd ~/Code/Research/computer_vision/EvaluateAD/APSIPA2024_AgriculturalAD/crop_data
python move.py --f 0.75

# Training EfficientAD and creating anomaly maps
cd ~/Code/Research/computer_vision/EfficientAD-modify
bash run.sh

# Run AD models in Anomalib
cd ~/Code/Research/computer_vision/EvaluateAD/APSIPA2024_AgriculturalAD/
bash run.sh

# Move obects selected by IoU to the target folder
cd ~/Code/Research/computer_vision/EvaluateAD/APSIPA2024_AgriculturalAD/crop_data
python move.py --f 0.9

# Training EfficientAD and creating anomaly maps
cd ~/Code/Research/computer_vision/EfficientAD-modify
bash run.sh

# Run AD models in Anomalib
cd ~/Code/Research/computer_vision/EvaluateAD/APSIPA2024_AgriculturalAD/
bash run.sh
