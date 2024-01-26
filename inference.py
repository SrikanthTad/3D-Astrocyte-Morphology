import os
import cv2
import pdb
import json
from tqdm import tqdm
from detectron2.config import get_cfg
from torchvision.utils import save_image
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from celldet_utils import setup_dataset, setup_oib_dataset

# Set your model configuration file and weights file
model_name = "CellDet_FasterRCNN_Base"
model_dir = os.path.join("saved_models", model_name)

input_type = "oib" # dataset, oib [It decides the operation mode]
dataset_name = "CellDet-OIB-Dataset-AllSlices2/validation"
# dataset_name = "CellDet-Dataset-AllSlices2/validation"
output_folder = "data/slices/half/all_slices2_done_predictions"
oib_path = "data/slices/half/all_slices2_done/cup1_20x_1.oib"
draw_preds = False
brightness_normalizer = 900. # Needs to be set to 4095.

anns_dir = os.path.join(output_folder, "annotations")
config_file = os.path.join(model_dir, "config.yaml")
weights_file = os.path.join(model_dir, "model_final.pth")

# Create a Detectron2 configuration
cfg = get_cfg()
cfg.merge_from_file(config_file)
cfg.MODEL.WEIGHTS = weights_file
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a confidence threshold for predictions
cfg.DATASETS.TRAIN = []
cfg.DATASETS.TEST = [dataset_name]
cfg.TEST.DETECTIONS_PER_IMAGE = 200

# Register your dataset
datasets_dir = "datasets/"
if input_type == 'dataset':
    setup_dataset(cfg, datasets_dir)
elif input_type == 'oib':
    setup_oib_dataset(cfg, oib_path, datasets_dir, brightness_normalizer=brightness_normalizer)
else:
    raise NotImplementedError

# Create a predictor
predictor = DefaultPredictor(cfg)

# Load your dataset and get its metadata
dataset_dicts = DatasetCatalog.get(dataset_name)
metadata = MetadataCatalog.get(dataset_name)

# Make the output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(anns_dir, exist_ok=True)

# Loop through the dataset and run inference
print(f"Running inference on {len(dataset_dicts)} images")
if draw_preds:
    print(f"Saving images with visualized predicted bounding boxes [NOTE: you need to disable this in order to run group_3d.py, change draw_preds to False]")
else:
    print(f"Saving images without predicted bounding box visualizations")
for d in tqdm(dataset_dicts):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    
    # Collect the required data for the json files
    file_name = os.path.join(anns_dir, d['file_name'].split('/')[-1].split('.')[0] + ".json")
    imagePath = f"../{d['file_name'].split('/')[-1]}"
    imageHeight = d['height']
    imageWidth = d['width']
    shapes = []
    for bbox in outputs['instances'].pred_boxes:
        bbox = bbox.tolist()
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
        points_dict = {
            "label": "astrocyte",
            "points": points,
        }
        shapes.append(points_dict)
    
    # Construct the json file 
    json_info = {
        "imagePath": imagePath,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth,
        "shapes": shapes
    }

    # Save the json file
    save_path = file_name
    with open(save_path, 'w') as f:
        json.dump(json_info, f)

    # Visualize the predictions (optional)
    if draw_preds:
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(output_folder, os.path.basename(d["file_name"])), v.get_image()[:, :, ::-1])
    else:
        cv2.imwrite(os.path.join(output_folder, os.path.basename(d["file_name"])), im)

print("Inference finished. Results saved in:", output_folder)
