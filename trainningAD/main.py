from pathlib import Path
import argparse
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import anomalib
from anomalib.data import PredictDataset, MVTec
from anomalib.engine import Engine
from anomalib.models import Fastflow, EfficientAd, Patchcore, Ganomaly, Draem, Padim
from anomalib.utils.post_processing import superimpose_anomaly_map
from anomalib import TaskType

# Launch arguments
parser = argparse.ArgumentParser(description="Agricultural Anomaly Detection")
parser.add_argument("--cls", type=str, default="bottle", help="Class name")
parser.add_argument("--model", type=str, default="fastflow", help="Model name")
parser.add_argument("--root", type=str, default="datasets/MVTec", help="Dataset root")
parser.add_argument("--task", type=str, default="SEGMENTATION", help="Task type")
parser.add_argument("--img_size", type=int, default=256, help="Image size")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
parser.add_argument("--test_id", type=int, default=0000, help="Test id")
args = parser.parse_args()

print("Arguments: ", args)

if args.task.upper() == "SEGMENTATION":
    task = TaskType.SEGMENTATION
elif args.task.upper() == "CLASSIFICATION":
    task = TaskType.CLASSIFICATION
elif args.task.upper() == "DETECTION":
    task = TaskType.DETECTION
else:
    raise ValueError("Task type not supported")


datamodule = MVTec(
    root= args.root,
    category= args.cls,
    train_batch_size= args.batch_size,
    eval_batch_size= args.batch_size,
    num_workers= args.num_workers,
    task=task,
    image_size=(args.img_size, args.img_size),
)

if args.model == "fastflow":
    model =  Fastflow(backbone="resnet18", flow_steps=8)
elif args.model == "efficientad":
    model =  EfficientAd(padding=1)
elif args.model == "patchcore":
    model =  Patchcore(backbone="wide_resnet50_2")
elif args.model == "ganomaly":
    model =  Ganomaly()
elif args.model == "draem":
    model = Draem(
    enable_sspcab=True,
    sspcab_lambda=0.2,
    beta=0.5,
    )
elif args.model == "padim":
    model = Padim()


callbacks = [
    ModelCheckpoint(
        mode="max",
        monitor="pixel_AUROC",
    ),
    EarlyStopping(
        monitor="pixel_AUROC",
        mode="max",
        patience=3,
    ),
]

engine = Engine(
    callbacks=callbacks,
    pixel_metrics="AUROC",
    accelerator="gpu",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
    devices=1,
    logger=False,
)

engine.fit(datamodule=datamodule, model=model)

engine.test(datamodule=datamodule, model=model)

inference_dataset = PredictDataset(path=args.root + f"/{args.cls}/test/tear/{args.test_id}.png")
inference_dataloader = DataLoader(dataset=inference_dataset, num_workers=47)

predictions = engine.predict(model=model, dataloaders=inference_dataloader)[0]

print(
    f'Image Shape: {predictions["image"].shape},\n'
    f'Anomaly Map Shape: {predictions["anomaly_maps"].shape}, \n'
    f'Predicted Mask Shape: {predictions["pred_masks"].shape}',
)

image_path = predictions["image_path"][0]
image_size = predictions["image"].shape[-2:]
image = np.array(Image.open(image_path).resize(image_size))

plt.imshow(image)
plt.savefig(f"{args.cls}_image.png")
plt.show()
import cv2

cv2.imwrite(f"{args.cls}_predict.png", image)

anomaly_map = predictions["anomaly_maps"][0]
anomaly_map = anomaly_map.cpu().numpy().squeeze()
plt.savefig(f"{args.cls}_anomaly_map.png")
plt.imshow(anomaly_map)
plt.savefig(f"{args.cls}_anomaly_map_1.png")
# plt.show()

heat_map = superimpose_anomaly_map(anomaly_map=anomaly_map, image=image, normalize=True)
plt.savefig(f"{args.cls}_heat_map.png")
plt.imshow(heat_map)
plt.savefig(f"{args.cls}_heat_map_1.png")
# plt.show()

pred_score = predictions["pred_scores"][0]
pred_labels = predictions["pred_labels"][0]
print(pred_score, pred_labels)

pred_masks = predictions["pred_masks"][0].squeeze().cpu().numpy()
plt.savefig(f"{args.cls}_pred_masks.png")
plt.imshow(pred_masks)
plt.savefig(f"{args.cls}_pred_masks_1.png")
# plt.show()