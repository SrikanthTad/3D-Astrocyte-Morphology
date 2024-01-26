import os
import pdb
import cv2
import math
import yaml
import json
import glob
import torch
import argparse
import numpy as np
import seaborn as sns
from tqdm import tqdm
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from detectron2.config import get_cfg
from yacs.config import CfgNode as CN
from torchvision.utils import save_image
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from celldet_utils import (
    setup_oib_dataset, 
    get_iou, 
    spectral_clustering,
    retrieve_cluster,
    bbox2frustum,
    postprocess_frustums,
    get_segmentation_nuclei,
    save_cropped_astro_masks,
    save_nuclei_from_oib,
    construct_slices_tensor,
    output2file,
    bfs2d,
    bfs3d,
    denoise_maskconn,
    plot_frustum_with_branches,
    find_closest_branch,
    compute_k_core,
    save_color_legend,
    save_total_colored_image,
    SegmentationCell,
    NucleiTypes
)

# Logger
from logger import get_logger
logger = get_logger("Logger")

def get_parser():
    parser = argparse.ArgumentParser(description="Astrocyte and nuclei detection from an OIB file")
    parser.add_argument("--config-file", default="configs/Configs.yaml", help="Path to the configuration file")
    return parser

def setup(args):
    # Setup the general configs
    config_file = args.config_file

    # Perform checks
    assert isinstance(config_file, str), f"Incorrect config file path. Expected string, instead got {config_file}."
    assert os.path.exists(config_file), f"Supplied config file {config_file} does not exist."

    # Load the config file
    with open(config_file, "r") as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error(exc)
    cfg = CN(cfg)

    # Print
    logger.info(f"Loaded the following configurations from {config_file}:\n{cfg}")

    # Setup the Detectron2 configs
    model_dir = os.path.join(cfg.MODELS_DIR, cfg.DETECTOR.NAME)
    d2config_file = os.path.join(model_dir, "config.yaml")
    weights_file = os.path.join(model_dir, "model_final.pth")
    assert os.path.exists(d2config_file), f"Expected a model config file at {d2config_file}, but did not find it."
    assert os.path.exists(weights_file), f"Expected model weights at {weights_file}, but did not find it."
    d2cfg = get_cfg()
    d2cfg.merge_from_file(d2config_file)
    d2cfg.MODEL.WEIGHTS = weights_file
    d2cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a confidence threshold for predictions
    d2cfg.DATASETS.TRAIN = []
    d2cfg.TEST.DETECTIONS_PER_IMAGE = 200
    dataset_name = '.'.join(cfg.DATA.OIB_PATH.split('/')[-1].split('.')[:-1])
    d2cfg.DATASETS.TEST = [dataset_name]

    return cfg, d2cfg

def setup_datasets(cfg, d2cfg):
    # Astrocyte dataset
    logger.info("Setting up the astrocytes dataset")
    oib_path = cfg.DATA.OIB_PATH
    datasets_dir = cfg.DATASETS_DIR
    brightness_normalizer = cfg.DATA.BRIGHTNESS_NORMALIZER
    setup_oib_dataset(d2cfg, oib_path, datasets_dir, brightness_normalizer=brightness_normalizer)

    # Nuclei dataset
    # TODO

def run_inference(cfg, d2cfg, params):
    # Create the predictor
    predictor = DefaultPredictor(d2cfg)

    # Load the astrocytes dataset and get its metadata
    dataset_name = d2cfg.DATASETS.TEST[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # Prepare the output directory
    output_dir = os.path.join(cfg.OUTPUT_DIR, f"{params['dir_counter']:03}_astrocyte_slices")
    params['dir_counter'] += 1
    preds_dir = os.path.join(cfg.OUTPUT_DIR, f"{params['dir_counter']:03}_detector_predictions")
    params['dir_counter'] += 1
    anns_dir = os.path.join(output_dir, "annotations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(anns_dir, exist_ok=True)

    # Run inference
    logger.info(f"Running inference on {len(dataset_dicts)} images")
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
        
        # ----------->001<----------- #
        # Save the images
        cv2.imwrite(os.path.join(output_dir, os.path.basename(d["file_name"])), im)

        # ----------->002<----------- #
        # Save with the bounding boxes
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(preds_dir, os.path.basename(d["file_name"])), v.get_image()[:, :, ::-1])
    
    logger.info(f"Inference finished. Images have been saved to {anns_dir}")

    # Construct the return parameters
    params["detector_predictions_dir"] = output_dir

    return params

def get_frustums(cfg, params):
    def visualize_bboxes_per_image(output_dir, images_list, bbox_list):
        # Label the bounding boxes with different colors
        sns.set_palette(palette='deep', n_colors=n_clusters)
        colors = sns.color_palette() # Create a list of RGB colors, where each color corresponds to a cluster
        assert len(bbox_list) == len(cluster_assignments) == n_boxes
        for idx in range(n_boxes):
            bbox_list[idx]['cluster'] = cluster_assignments[idx]
            bbox_list[idx]['color'] = colors[cluster_assignments[idx]]

        # Visualize the bounding boxes per image
        n_images = len(images_list)
        save_dir = os.path.join(output_dir, f"{params['dir_counter']:03}_bounding_boxes_per_image")
        params['dir_counter'] += 1
        os.makedirs(save_dir, exist_ok=True)
        logger.info("Loading images and visualizing the clusters")
        for idx, image_path in enumerate(images_list):
            # Fetch image information
            z = image_path.split('/')[-1].split('.')[0]
            bboxes = []
            anns_path = os.path.join(anns_dir, f"{z}.json")
            assert os.path.exists(anns_path)

            # Collect all the corresponding bounding boxes
            for bbox_info in bbox_list:
                if bbox_info['annotations_file'] == anns_path:
                    bboxes.append(bbox_info)
            logger.info(f"[{idx + 1}/{n_images}] Collected {len(bboxes)} bounding boxes for image {image_path}")

            # Load the image
            image = Image.open(image_path).convert('RGB')

            # Draw the bounding boxes with the corresponding colors
            draw = ImageDraw.Draw(image)
            for bbox in bboxes:
                x1 = bbox['x1']
                y1 = bbox['y1']
                x2 = bbox['x2']
                y2 = bbox['y2']
                outline_color = tuple([int(x * 255) for x in bbox['color']])
                draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], outline=outline_color)
            
            # Save the labeled image
            save_path = os.path.join(save_dir, image_path.split('/')[-1])
            image.save(save_path)

    def visualize_frustum_slices(save_dir, n_clusters, images_list, anns_dir):
        # Visualize the frustums per cluster
        sns.set_palette(palette='deep', n_colors=n_clusters)
        colors = sns.color_palette() # Create a list of RGB colors, where each color corresponds to a cluster
        for cluster_idx in tqdm(range(n_clusters)):
            os.makedirs(os.path.join(save_dir, str(cluster_idx)), exist_ok=True)
            for idx, image_path in enumerate(images_list):
                # Fetch image information
                z = image_path.split('/')[-1].split('.')[0]
                bboxes = []
                anns_path = os.path.join(anns_dir, f"{z}.json")
                assert os.path.exists(anns_path)

                # Collect all the corresponding bounding boxes
                for bbox_info in frustums_list:
                    if bbox_info['cluster'] == cluster_idx:
                        bboxes.append(bbox_info)

                # Load the image
                image = Image.open(image_path).convert('RGB')
                
                # Draw the bounding boxes with the corresponding colors
                draw = ImageDraw.Draw(image)
                for bbox in bboxes:
                    x1 = bbox['x1']
                    y1 = bbox['y1']
                    x2 = bbox['x2']
                    y2 = bbox['y2']
                    z1 = bbox['z1']
                    z2 = bbox['z2']
                    if z1 <= float(z) <= z2:
                        outline_color = tuple([int(x * 255) for x in colors[cluster_idx]])
                        draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], outline=outline_color)
                
                # Save the labeled image
                save_path = os.path.join(save_dir, str(cluster_idx), image_path.split('/')[-1])
                image.save(save_path)

    # Load the data
    output_dir = cfg.OUTPUT_DIR
    images_dir = params['detector_predictions_dir']
    anns_dir = os.path.join(images_dir, "annotations")
    images_list = sorted(glob.glob(images_dir + "/*.png"))
    anns_list = sorted(glob.glob(anns_dir + "/*.json"))
    assert len(images_list) == len(anns_list), f"Different number of images and annotations files. Found {len(images_list)} images and {len(anns_list)} annotation files."
    logger.info(f"Detected {len(images_list)} images.\nDetected {len(anns_list)} annotation files.")
    
    # Collect the bounding boxes along with their z values
    bbox_list = []
    for anns_path in tqdm(anns_list):
        # Load the annotations file
        with open(anns_path, 'rb') as f:
            anns = json.load(f)

        # Extract the z-value
        z = float(anns_path.split('/')[-1].split('.')[0])
        
        # Build the bounding box information
        for label in anns['shapes']:
            points = label['points']
            bbox_info = {
                "x1": min(points[0][0], points[1][0]), # x min
                "x2": max(points[0][0], points[1][0]), # x max
                "y1": min(points[0][1], points[1][1]), # y min
                "y2": max(points[0][1], points[1][1]), # y max
                "z": z,
                "annotations_file": anns_path,
            }

            # Append to the list of bounding boxes
            bbox_list.append(bbox_info)
    logger.info(f"Extracted {len(bbox_list)} bounding boxes in total")

    # Compute the IoU distances between all pairs of bounding boxes [len(bbox_list) x len(bbox_list)]
    logger.info("Computing the adjacency matrix...")
    n_boxes = len(bbox_list)
    iou_matrix = torch.empty(size=(n_boxes, n_boxes))
    for i in tqdm(range(n_boxes)):
        for j in range(n_boxes):
            iou_score = get_iou(bbox_list[i], bbox_list[j])
            iou_matrix[i][j] = iou_score
    adjacency_matrix = np.array(iou_matrix)

    # Automatic inference of the number of frustums
    n_clusters = int(n_boxes / len(anns_list))
    logger.info(f"Approximating the data as {n_clusters} clusters per image")
    cluster_assignments, _ = spectral_clustering(adjacency_matrix, n_clusters)
    logger.info(f"Clustered the bounding boxes into {n_clusters} clusters.")

    # ----------->003<----------- #
    # Visualize the bounding boxes per image
    visualize_bboxes_per_image(output_dir, images_list, bbox_list)
    
    # For each cluster idx, retrieve its bounding boxes and compute the 3D frustum
    frustums_list = []
    logger.info(f"Retrieving {n_clusters} 3D frustums")
    for idx in tqdm(range(n_clusters)):
        # Retrieve the cluster
        cluster = retrieve_cluster(bbox_list, idx)
        
        # Deduce the frustum
        frustum = bbox2frustum(cluster)
        frustum['cluster'] = idx

        # Append to the list of frustums
        frustums_list.append(frustum)

    # Post-process frustums (e.g. remove outliers)
    # TODO: postprocess frustums
    frustums_list = postprocess_frustums(frustums_list)
    logger.info("Frustums have been identified!")

    # ----------->004<----------- #
    # Visualize and save the frustums
    save_dir = os.path.join(output_dir, f"{params['dir_counter']:03}_frustum_slices")
    params['dir_counter'] += 1
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "frustums_list.json"), 'w') as f:
        json.dump(frustums_list, f)
    if cfg.VISUALIZE_FRUSTUM_SLICES:
        logger.info("Visualizing the frustum slices")
        visualize_frustum_slices(save_dir, n_clusters, images_list, anns_dir)

    # Update params
    params['frustums_list'] = frustums_list

    return params

def run_frustum_segmentation(cfg, params):
    # Setup parameters
    frustums_list = params['frustums_list']
    n_frustums = len(frustums_list)
    images_dir = params['detector_predictions_dir']
    output_dir = cfg.OUTPUT_DIR

    # Load the images
    images_list = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    n_images = len(images_list)
    logger.info(f"Loaded {n_images} images from {images_dir}")

    # Crop the images
    cropped_images = []
    images_list_pil = []
    for image_path in tqdm(images_list):
        # Retrieve the image and its z-coordinate
        image = Image.open(image_path).convert('RGB')
        images_list_pil.append(image)
        z = float(image_path.split('/')[-1].split('.')[0])
        
        for frustum in frustums_list:
            within_frustum = frustum['z2'] >= z >= frustum['z1']
            
            # Crop the image if within the frustum z-range
            if within_frustum:
                x_min = frustum['x1']
                x_max = frustum['x2']
                y_min = frustum['y1']
                y_max = frustum['y2']
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                cropped_image_info = {
                    "x1": x_min,
                    "x2": x_max,
                    "y1": y_min,
                    "y2": y_max,
                    "z": z,
                    "image": cropped_image,
                    "frustum_id": frustum['cluster']
                }
                cropped_images.append(cropped_image_info)
    n_cropped_images = len(cropped_images)
    logger.info(f"Extracted {n_cropped_images} cropped images")

    # Initialize the astrocyte segmentation model
    NUCLEI_CLASS_MAP = {
        0: NucleiTypes.BACKGROUND,
        1: NucleiTypes.NUCLEUS,
        2: NucleiTypes.BORDER
    }
    NUCLEI_CLASSES = len(set(NUCLEI_CLASS_MAP.values()))
    nuclei_model_list = [
        {
            "architecture": cfg.SEGMENTATION.ASTROCYTES.ARCHITECTURE,
            "encoder_name": cfg.SEGMENTATION.ASTROCYTES.ENCODER_NAME,
            "weight": cfg.SEGMENTATION.ASTROCYTES.WEIGHTS
        }
    ]
    preprocessing_fn = get_preprocessing_fn('resnet50', pretrained='imagenet')
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    N_INPUT_CHANNEL = cfg.SEGMENTATION.N_INPUT_CHANNEL

    # Load astrocyte models (this can be 1 or 3 models)
    nuclei_models = [
        SegmentationCell.load_from_checkpoint(
            model_config["weight"],
            architecture=model_config["architecture"],
            encoder_name=model_config["encoder_name"],
            N_INPUT_CHANNEL=N_INPUT_CHANNEL,
            NUCLEI_CLASSES=NUCLEI_CLASSES
        ).to(device)
        for model_config in nuclei_model_list[:1]
    ]

    # Run through all images and extract segmentation masks of the astrocytes
    for idx in tqdm(range(n_cropped_images)):
        image = cropped_images[idx]

        # Construct the input
        image = torch.tensor(np.array(image['image']))
        mask = get_segmentation_nuclei(image, nuclei_models, preprocessing_fn)
        cropped_images[idx]['mask'] = mask
    
    # ----------->005<----------- #
    # Saving cropped masks and images
    cropped_astro_masks_save_dir = os.path.join(output_dir, f"{params['dir_counter']:03}_cropped_astrocyte_masks_per_frustum")
    params['dir_counter'] += 1
    logger.info(f"Saving cropped masks and images of astrocytes to {cropped_astro_masks_save_dir}")
    params['cropped_images'] = cropped_images
    save_cropped_astro_masks(cropped_images, cropped_astro_masks_save_dir)

    # Reconstruct the slice masks
    unique_zs = sorted(set([image_info['z'] for image_info in cropped_images]))
    logger.info(f"Identified {len(unique_zs)} unique z-coordinates")
    slice_masks = {unique_z: torch.zeros((800, 800, 1)) for unique_z in unique_zs}
    for cropped_image_info in tqdm(cropped_images):
        x_min = int(cropped_image_info['x1'])
        x_max = int(cropped_image_info['x2'])
        y_min = int(cropped_image_info['y1'])
        y_max = int(cropped_image_info['y2'])
        z = cropped_image_info['z']
        mask = cropped_image_info['mask']

        # Account for potential discrepancies in coordinates when putting the masks back
        if x_max - x_min != mask.shape[1] or y_max - y_min != mask.shape[0]:
            x_max = x_min + mask.shape[1]
            y_max = y_min + mask.shape[0]

        # Replace the mask
        slice_masks[z][y_min:y_max, x_min:x_max] = torch.tensor(mask)
    params['Slice Masks'] = slice_masks

    # ----------->006<----------- #
    # Save the reconstructed masks
    astrocyte_masks_dir = os.path.join(output_dir, f"{params['dir_counter']:03}_astrocyte_masks")
    params['dir_counter'] += 1
    os.makedirs(astrocyte_masks_dir, exist_ok=True)
    logger.info(f"Saving the masks to {astrocyte_masks_dir}")
    for k, v in slice_masks.items():
        save_path = os.path.join(astrocyte_masks_dir, f"mask_{str(int(k))}.png")
        save_image(v.permute(2, 0, 1) / 2., save_path) # 2 classes, hence divide by 2
    
    # Visualize the collapsed masks (including the colored collapsed mask)
    collapsed_mask = torch.max(torch.stack(list(slice_masks.values())), dim=0)[0]
    collapsed_image = torch.mean(torch.stack([torch.tensor(np.array(image) / 255.) for image in images_list_pil]), dim=0)
    branch_mask = (torch.sum((torch.stack(list(slice_masks.values())).squeeze(3) == 1).float(), dim=0) != 0).float()
    body_mask = (torch.sum((torch.stack(list(slice_masks.values())).squeeze(3) == 2).float(), dim=0) != 0).float()
    colored_branch_mask = torch.zeros(size=(branch_mask.shape[0], branch_mask.shape[1], 3))
    colored_body_mask = torch.zeros(size=(body_mask.shape[0], body_mask.shape[1], 3))
    branch_color = torch.tensor([1., 0., 0.])
    body_color = torch.tensor([0., 1., 0.])
    colored_branch_mask[[branch_mask == 1]] = branch_color
    colored_body_mask[[body_mask == 1]] = body_color
    collapsed_mask = (colored_branch_mask + colored_body_mask) / 2.
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(collapsed_image)
    ax[0].set_title('Collapsed images')
    ax[1].imshow(collapsed_mask)
    ax[1].set_title('Collapsed masks')
    for ax_ in ax:
        ax_.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(astrocyte_masks_dir, "collapsed_masks.png"), dpi=300)
    plt.close(fig)
    del fig, ax
    params['Collapsed Astrocyte Mask'] = collapsed_mask

    # Extract the nuclei masks
    nuclei_model_list = [
        {
            "architecture": cfg.SEGMENTATION.NUCLEI.ARCHITECTURE,
            "encoder_name": cfg.SEGMENTATION.NUCLEI.ENCODER_NAME,
            "weight": cfg.SEGMENTATION.NUCLEI.WEIGHTS
        }
    ]
    nuclei_models = [
        SegmentationCell.load_from_checkpoint(
            model_config["weight"],
            architecture=model_config["architecture"],
            encoder_name=model_config["encoder_name"]
        ).to(device)
        for model_config in nuclei_model_list[:1]
    ]

    # ----------->007<----------- #
    # Slice the OIB file to save the nuclei channel
    nuclei_images_save_dir = os.path.join(output_dir, f"{params['dir_counter']:03}_all_slices2_done_nuclei")
    params['dir_counter'] += 1
    save_nuclei_from_oib(oib_path=cfg.DATA.OIB_PATH, save_dir=nuclei_images_save_dir, logger=logger)

    # Load the images of nuclei
    nuclei_images_list = sorted(glob.glob(os.path.join(nuclei_images_save_dir, "*.png")))
    nuclei_images_dict = {float(nuclei_image_path.split('/')[-1].split('.')[0]): np.array(Image.open(nuclei_image_path)) / 255. for nuclei_image_path in nuclei_images_list}
    logger.info(f"Loaded {len(nuclei_images_dict)} images of nuclei slices")

    # Initialize empty nuclei masks
    nuclei_masks = {k: torch.zeros((v.shape[0], v.shape[1], 1)) for k, v in nuclei_images_dict.items()}

    # For each frustum, run segmentation on the corresponding part of the nuclei slices
    logger.info(f"Running segmentation on cropped nuclei images for {len(cropped_images)} frustums")
    for idx in tqdm(range(n_cropped_images)):
        cropped_image_info = cropped_images[idx]
        x1 = int(cropped_image_info['x1'])
        x2 = int(cropped_image_info['x2'])
        y1 = int(cropped_image_info['y1'])
        y2 = int(cropped_image_info['y2'])
        z = cropped_image_info['z']
        w, h = cropped_image_info['image'].size

        # Select the corresponding nuclei image
        nuclei_image = torch.tensor(nuclei_images_dict[z])
        cropped_nuclei_image = nuclei_image[y1:y1 + h, x1:x1 + w]
        
        # Replicate the green channel in the red and blue channels
        cropped_nuclei_image[..., 0] = cropped_nuclei_image[..., 1]
        cropped_nuclei_image[..., 2] = cropped_nuclei_image[..., 1]

        # Run segmentation
        cropped_nuclei_image = cropped_nuclei_image * 255.
        output = get_segmentation_nuclei(cropped_nuclei_image, nuclei_models, preprocessing_fn)

        # Save the mask
        cropped_images[idx]['nuclei-mask'] = output
        nuclei_masks[z][y1:y1 + h, x1:x1 + w] = torch.tensor(output)
    
    # Save the colored collapsed nuclei mask
    stacked_nuclei_masks = torch.stack([nuclei_mask for nuclei_mask in nuclei_masks.values()], axis=2).squeeze(3)
    collapsed_nuclei_masks = torch.max(stacked_nuclei_masks, axis=2)[0]
    colored_collapsed_nuclei_masks = torch.zeros((collapsed_nuclei_masks.shape[0], collapsed_nuclei_masks.shape[1], 3))
    nuclei_color = torch.tensor([0., 0., 1.])
    boundary_color = torch.tensor([0., 1., 0.])
    colored_collapsed_nuclei_masks[[collapsed_nuclei_masks == 1]] = nuclei_color
    colored_collapsed_nuclei_masks[[collapsed_nuclei_masks == 2]] = boundary_color
    params["Collapsed Nuclei Mask"] = colored_collapsed_nuclei_masks
    
    # Record the nuclei masks to the frustums list
    for idx in range(n_frustums):
        frustum_idx = frustums_list[idx]['cluster']

        # Collect all slices corresponding to this frustum ID
        slices_per_frustum = {cropped_image['z']: torch.tensor(cropped_image['nuclei-mask']).float() for cropped_image in cropped_images if cropped_image['frustum_id'] == frustum_idx}
        slices_per_frustum = dict(sorted(slices_per_frustum.items())) # Sort the dictionary by the z coordinate
        slice_3d_frustum = construct_slices_tensor(slices_per_frustum).squeeze(3) # Squeeze the last channel

        # Assign the 3D mask to the frustum
        frustums_list[frustum_idx]['nuclei-mask3d'] = slice_3d_frustum
    
    # ----------->008<----------- #
    # Save the nuclei masks
    nuclei_zoomed_segmentation_save_dir = os.path.join(output_dir, f"{params['dir_counter']:03}_nuclei_zoomed_masks")
    params['dir_counter'] += 1
    os.makedirs(nuclei_zoomed_segmentation_save_dir, exist_ok=True)
    for idx, (k, v) in enumerate(nuclei_masks.items()):
        save_path = os.path.join(nuclei_zoomed_segmentation_save_dir, f"{idx:03}.png")
        save_image(v.permute(2, 0, 1) / 2., save_path)
        
        # Color the output
        output = np.array(v)
        colored_output = np.repeat(output, repeats=3, axis=-1)
        colored_output[np.squeeze(output, axis=-1)==0] = np.array([0., 0., 0.])
        colored_output[np.squeeze(output, axis=-1)==1] = np.array([0., 0., 1.])
        colored_output[np.squeeze(output, axis=-1)==2] = np.array([0., 1., 0.])

        # Save the colored output
        save_path = os.path.join(nuclei_zoomed_segmentation_save_dir, f"colored_{idx:03}.png")
        output2file(colored_output, 1., save_path)

    # TODO: save collapsed nuclei mask

    # 3D BFS  
    # Construct a 3D tensor for each frustum
    logger.info("Constructing 3D masks for all frustums")
    for idx in range(n_frustums):
        frustum_idx = frustums_list[idx]['cluster']

        # Collect all slices corresponding to this frustum ID
        slices_per_frustum = {cropped_image['z']: torch.tensor(cropped_image['mask']).float() for cropped_image in cropped_images if cropped_image['frustum_id'] == frustum_idx}
        slices_per_frustum = dict(sorted(slices_per_frustum.items())) # Sort the dictionary by the z coordinate
        slice_3d_frustum = construct_slices_tensor(slices_per_frustum).squeeze(3) # Squeeze the last channel

        # Assign the 3D mask to the frustum
        frustums_list[frustum_idx]['mask3d'] = slice_3d_frustum
    
    # Perform the connectivity check for all 3D masks and run K-core if needed
    enable_kcore = cfg.SEGMENTATION.KCORE.ENABLE
    k_core_value = cfg.SEGMENTATION.KCORE.K
    if enable_kcore:
        logger.info(f"Using the K-core with k={k_core_value}")
    for frustum_idx in tqdm(range(n_frustums)):
        if enable_kcore:
            frustums_list[frustum_idx]['mask3d'] = compute_k_core(frustums_list[frustum_idx]['mask3d'], k_core_value)
        mask3d = frustums_list[frustum_idx]['mask3d']
        connectivity_mask = bfs3d(mask3d)
        frustums_list[frustum_idx]['maskconn'] = connectivity_mask
    
    # Denoise the 3D mask, by removing branch pixels that do not belong to the biggest branch cluster
    noise_frac = []
    logger.info("Denoising the 3D masks")
    for frustum_idx in tqdm(range(n_frustums)):
        frustum_info = frustums_list[frustum_idx]

        frustums_list[frustum_idx]['maskconn-denoised'], noise_mask = denoise_maskconn(frustum_info['mask3d'], frustum_info['maskconn'])
        noise_frac.append(1 - noise_mask.sum().item() / noise_mask.numel())
    # logger.info(f"Noise fractions: {noise_frac}")

    # Run 2D BFS on each branch mask
    logger.info("Identifying the branches")
    branches_save_dir = os.path.join(output_dir, f"{params['dir_counter']:03}_colored_branches_per_frustum")
    params['dir_counter'] += 1
    min_pixels_per_branch = cfg.SEGMENTATION.MIN_PIXELS_PER_BRANCH
    for frustum_idx in tqdm(range(n_frustums)):
        maskconn_denoised_collapsed = torch.max(frustums_list[frustum_idx]['maskconn-denoised'], dim=0)
        branch_only = (maskconn_denoised_collapsed[0] == 1.0).float()
        collapsed_branches = bfs2d(branch_only)
        
        # Remove very small branches
        for i in range(maskconn_denoised_collapsed[0].shape[0]):
            for j in range(maskconn_denoised_collapsed[0].shape[1]):
                if maskconn_denoised_collapsed[0][i][j] == 0.:
                    bg_idx = collapsed_branches[i][j].item() # Identify the idx corresponding to the background
                    break
        branch_idxs = collapsed_branches.unique(return_counts=True)[0]
        branch_count = collapsed_branches.unique(return_counts=True)[1]
        for branch_idx, branch_pix_count in zip(branch_idxs, branch_count):
            branch_idx = branch_idx.item()
            branch_pix_count = branch_pix_count.item()
            if branch_pix_count < min_pixels_per_branch:
                branch_idx_mask = (collapsed_branches == branch_idx).float()
                collapsed_branches = collapsed_branches * (1 - branch_idx_mask)
        
        # # Save the branch segmentations
        # save_image(collapsed_branches / collapsed_branches.max(), f"results/collapsed_branches_{frustum_idx:03}.png")

        # Record the denoised branches
        frustums_list[frustum_idx]['collapsed-branches-idxs'] = collapsed_branches
        denoised_branch_idxs = collapsed_branches.unique(return_counts=True)[0]
        denoised_branch_count = collapsed_branches.unique(return_counts=True)[1]
        n_branches = len(denoised_branch_idxs) - 1

        # De-collapse the branches to color the whole frustum
        # First, re-format the collapsed mask such that 0 corresponds to the background
        idx2refidx = dict(zip(denoised_branch_idxs[denoised_branch_idxs != bg_idx].tolist(), list(range(1, n_branches + 1))))
        ref_collapsed_branches = torch.zeros(collapsed_branches.shape)
        for i in range(collapsed_branches.shape[0]):
            for j in range(collapsed_branches.shape[1]):
                if collapsed_branches[i][j] == bg_idx:
                    ref_collapsed_branches[i][j] = 0.
                else:
                    ref_collapsed_branches[i][j] = idx2refidx[collapsed_branches[i][j].item()]
        
        # Second, re-introduce the collapsed body pixels
        # ref_collapsed_branches[maskconn_denoised_collapsed[0] == 2.] = -1
        
        # Assign classes to the branches
        mask3d = frustums_list[frustum_idx]['mask3d']
        branches3d = torch.zeros(mask3d.shape)
        for slice in range(mask3d.shape[0]):
            for row in range(mask3d.shape[1]):
                for col in range(mask3d.shape[2]):
                    if mask3d[slice][row][col] == 2.:
                        # If body
                        branches3d[slice][row][col] = -1
                    elif mask3d[slice][row][col] == 1.:
                        # If branch
                        if ref_collapsed_branches[row][col] > 0. and maskconn_denoised_collapsed[0][row][col] != 2.:
                            # If there is a branch ID assigned, then assign it to this pixel
                            branches3d[slice][row][col] = ref_collapsed_branches[row][col]
                        elif ref_collapsed_branches[row][col] == 0. and maskconn_denoised_collapsed[0][row][col] == 2.:
                            # If no branch ID is assigned but the collapsed masks thinks this is the body, then find the closest branch and assign its ID
                            closest_branch_id = find_closest_branch(ref_collapsed_branches, (row, col))
                            branches3d[slice][row][col] = closest_branch_id
                        else:
                            # Assign background
                            branches3d[slice][row][col] = 0
                    else:
                        # If background
                        branches3d[slice][row][col] = 0
        
        # ----------->009<----------- #
        # Third, plot the slices of the frustum and save them
        plot_frustum_with_branches(branches3d, branches_save_dir, frustum_idx)
        frustums_list[frustum_idx]['branches3d'] = branches3d

    # Perform the connectivity check for all 3D masks
    logger.info("Identifying the corresponding nuclei")
    for frustum_idx in tqdm(range(n_frustums)):
        mask3d = frustums_list[frustum_idx]['nuclei-mask3d']
        maskconn = bfs3d((frustums_list[frustum_idx]['nuclei-mask3d'] == 1.).float())

        # Identify which category each cluster refers to
        mc_classes = {"Background": [], "Nuclei": []}
        recorded_idxs = []
        for i in range(mask3d.shape[0]):
            for j in range(mask3d.shape[1]):
                for k in range(mask3d.shape[2]):
                    mask3d_val = mask3d[i][j][k].item()
                    mc_val = maskconn[i][j][k].item()
                    if mc_val not in recorded_idxs:
                        if mask3d_val == 1.:
                            mc_classes['Nuclei'].append(mc_val)
                        else:
                            mc_classes['Background'].append(mc_val)
                        recorded_idxs.append(mc_val)
        
        # Construct the reformatted 3D mask where 0 corresponds to the background
        ref_maskconn = torch.zeros(maskconn.shape)
        for idx, nuclei_cluster_idx in enumerate(mc_classes['Nuclei']):
            ref_maskconn[maskconn == nuclei_cluster_idx] = idx
        frustums_list[frustum_idx]['nuclei-maskconn'] = ref_maskconn

        # Find the cluster with the largest overlap with the clean astrocyte 3D mask
        astro_mask3d = frustums_list[frustum_idx]['mask3d']
        n_nuclei_clusters = len(mc_classes['Nuclei'])
        overlap_scores = {nuclei_cluster_idx: 0 for nuclei_cluster_idx in range(1, n_nuclei_clusters + 1)}
        if len(overlap_scores) > 0:
            for nuclei_cluster_idx in overlap_scores.keys():
                overlap_score = ((ref_maskconn == float(nuclei_cluster_idx)).float() * (astro_mask3d == 2.).float()).sum()
                overlap_scores[nuclei_cluster_idx] = overlap_score.item()

            # Select the nucleus with the highest overlap
            native_nucleus_idx = max(overlap_scores, key=lambda k: overlap_scores[k])
            nucleus3d = (ref_maskconn == native_nucleus_idx).float()
        else:
            nucleus3d = torch.zeros(ref_maskconn.shape)
        frustums_list[frustum_idx]['nucleus3d'] = nucleus3d

    # ----------->010<----------- #
    # Plot the body + branches + nucleus
    final_plots_save_dir = os.path.join(output_dir, f"{params['dir_counter']:03}_final_plots_per_frustum")
    params['dir_counter'] += 1
    logger.info("Plotting the final results")
    total_colored_image3d = torch.zeros(size=(len(params['Slice Masks']), params['Slice Masks'][0].shape[0], params['Slice Masks'][0].shape[1], 3))
    for frustum_idx in tqdm(range(n_frustums)):
        frustum = frustums_list[frustum_idx]
        branches3d = frustum['branches3d']
        nucleus3d = frustum['nucleus3d']

        # Construct the final 3D image
        mask_idxs = range(len(branches3d.unique()) + len(nucleus3d.unique()) - 1) # background, branch 1, branch 2, ..., nucleus, body
        image3d = torch.zeros(size=(branches3d.shape[0], branches3d.shape[1], branches3d.shape[2], len(mask_idxs))) # -1 because background is calculated twice

        # Add the branch indices
        for branch_idx in branches3d.unique():
            if branch_idx == -1:
                # Body
                image3d[branches3d == branch_idx] = mask_idxs[-2]
            elif branch_idx > 0:
                # Branch
                image3d[branches3d == branch_idx] = branch_idx
            else:
                # Background
                pass
        image3d[nucleus3d == 1] = mask_idxs[-1]
        frustums_list[frustum_idx]['combined3d'] = image3d
        
        # Plot each slice
        colored_image3d = torch.zeros(size=(branches3d.shape[0], branches3d.shape[1], branches3d.shape[2], 3))
        n_colors = len(branches3d.unique()[branches3d.unique() > 0]) + 2 # color the branches, the nucleus and the body
        colors = torch.tensor(sns.color_palette('deep', n_colors=n_colors))
        
        # Color the branches
        for idx, branch_idx in enumerate(branches3d.unique()[branches3d.unique() > 0]):
            colored_image3d[branches3d == branch_idx] = colors[idx + 2] # branches
            colored_image3d[(branches3d == branch_idx) * (nucleus3d == 1)] = 0.5 * (colors[idx + 2] + colors[1]) # branches and nuclei overlap

        # Color the nucleus and the body
        colored_image3d[nucleus3d == 1] = colors[1] # nuclei
        colored_image3d[branches3d == -1] = colors[0] # body
        colored_image3d[(nucleus3d == 1) * (branches3d == -1)] = 0.5 * (colors[0] + colors[1]) # nuclei and body overlap
        frustums_list[frustum_idx]['colored_3d'] = colored_image3d

        # Save each slice
        save_dir = os.path.join(final_plots_save_dir, str(frustum_idx))
        os.makedirs(save_dir, exist_ok=True)
        for idx, slice in enumerate(colored_image3d):
            save_path = os.path.join(save_dir, f"slice_{idx:03}.png")
            save_image(slice.permute(2, 0, 1), save_path)

        # Update the total image
        x1 = int(frustum['x1'])
        y1 = int(frustum['y1'])
        z1 = int(frustum['z1'])
        x2 = x1 + colored_image3d.shape[2]
        y2 = y1 + colored_image3d.shape[1]
        z2 = z1 + colored_image3d.shape[0]
        try:
            total_colored_image3d[z1:z2, y1:y2, x1:x2, :] = colored_image3d.clone()
        except Exception as e:
            pdb.set_trace()

        # Save the color legend
        body_color = colors[0]
        nuclei_color = colors[1]
        save_color_legend(final_plots_save_dir, body_color, nuclei_color)

    # Save the total final image
    save_total_colored_image(final_plots_save_dir, total_colored_image3d)

    return params

def compute_stats(cfg, params):
    frustums_list = params['frustums_list']
    cropped_images = params['cropped_images']
    n_frustums = len(frustums_list)
    output_dir = cfg.OUTPUT_DIR
    n_pixels_branch = []
    n_pixels_body = []
    n_pixels_nucleus = []

    # Collect the data
    for frustum_idx in tqdm(range(n_frustums)):
        frustum = frustums_list[frustum_idx]
        branches3d = frustum['branches3d']
        nucleus3d = frustum['nucleus3d']

        # Compute the required numbers
        cell_surface = 0        # number of astrocyte (body+branches) pixels in 3D in this frustum
        primary_branches = 0    # number of unique branches in 3D in this frustum
        nuclei_surface = 0      # number of nuclei pixels in 3D in this frustum
        number_nuclei = 0       # number of nuclei for this astrocyte (NOTE: at most 1!)
        number_astrocytes = 1   # always 1 astrocyte per frustum

        # Astrocyte channel
        for idx, count in zip(branches3d.unique(return_counts=True)[0], branches3d.unique(return_counts=True)[1]):
            idx = idx.item()
            count = count.item()
            
            # Check what the index corresponds to
            if idx == -1:
                # Body
                n_pixels_body.append(count)
                cell_surface += count
            elif idx > 0:
                # Branch
                n_pixels_branch.append(count)
                cell_surface += count
                primary_branches += 1
            else:
                # Background
                pass
        
        # Nuclei channel
        for idx, count in zip(nucleus3d.unique(return_counts=True)[0], nucleus3d.unique(return_counts=True)[1]):
            idx = idx.item()
            count = count.item()
            
            # Check what the index corresponds to
            if idx > 0:
                # Nuclei
                n_pixels_nucleus.append(count)
                nuclei_surface += count
                number_nuclei += 1
            else:
                # Background
                pass
        
        # Save the results in the frustums list
        frustums_list[frustum_idx]['Cell Surface'] = cell_surface
        frustums_list[frustum_idx]['Primary Branches'] = primary_branches
        frustums_list[frustum_idx]['Nuclei Surface'] = nuclei_surface
        frustums_list[frustum_idx]['Number of Nuclei'] = number_nuclei
        frustums_list[frustum_idx]['Number of Astrocytes'] = number_astrocytes

    # Compute the total statistics
    total_cell_surface = sum([frustum['Cell Surface'] for frustum in frustums_list]) # number of astrocyte (body+branches) pixels in 3D
    total_primary_branches = sum([frustum['Primary Branches'] for frustum in frustums_list]) # number of unique branches in 3D
    total_nuclei_surface = sum([frustum['Nuclei Surface'] for frustum in frustums_list]) # number of nuclei pixels in 3D
    total_number_nuclei = sum([frustum['Number of Nuclei'] for frustum in frustums_list]) # number of nuclei for the entire 3D scene
    total_number_astrocytes = n_frustums # number of astrocytes for the entire 3D scene

    stats = {
        'body': {
            'mean': np.mean(n_pixels_body),
            'std': math.sqrt(np.var(n_pixels_body)),
            'n_samples': len(n_pixels_body)
        },
        'branches': {
            'mean': np.mean(n_pixels_branch),
            'std': math.sqrt(np.var(n_pixels_branch)),
            'n_samples': len(n_pixels_branch)
        },
        'nucleus': {
            'mean': np.mean(n_pixels_nucleus),
            'std': math.sqrt(np.var(n_pixels_nucleus)),
            'n_samples': len(n_pixels_nucleus)
        }
    }
    # Print the general statistics
    stats_text = "3D Statistics:\n"
    for k, v in stats.items():
        mean = v['mean']
        std = v['std']
        n_samples = v['n_samples']
        stats_text += f"{k}: Mean={round(mean, 2)}, Standard Deviation={round(std, 2)}, Number of samples={n_samples}\n"
    logger.info(stats_text)

    # Plot the overall statistics
    stats_save_dir = os.path.join(output_dir, f"{params['dir_counter']:03}_statistics")
    os.makedirs(stats_save_dir, exist_ok=True)
    params['dir_counter'] += 1
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    # Plot distribution of Branches
    axs[0].hist(n_pixels_branch, bins=20, edgecolor='black')
    axs[0].set_title('Branches Distribution in 3D')
    axs[0].set_xlabel('Number of Pixels')
    axs[0].set_ylabel('Frequency')
    # Plot distribution of Body
    axs[1].hist(n_pixels_body, bins=20, edgecolor='black')
    axs[1].set_title('Body Distribution in 3D')
    axs[1].set_xlabel('Number of Pixels')
    axs[1].set_ylabel('Frequency')
    # Plot distribution of Nucleus
    axs[2].hist(n_pixels_nucleus, bins=20, edgecolor='black')
    axs[2].set_title('Nucleus Distribution in 3D')
    axs[2].set_xlabel('Number of Pixels')
    axs[2].set_ylabel('Frequency')
    fig.tight_layout()
    fig.savefig(os.path.join(stats_save_dir, "overall_statistics.png"), dpi=300)
    plt.close(fig)
    del fig, axs

    # Compute the statistics for each slice of each frustum
    logger.info("Computing statistics for each slice of each frustum")
    stats_save_dir_per_frustum = os.path.join(stats_save_dir, "per_frustum")
    os.makedirs(stats_save_dir_per_frustum, exist_ok=True)
    for frustum_idx in tqdm(range(n_frustums)):
        frustum_info = frustums_list[frustum_idx]
        branches3d = frustum_info['branches3d']
        nucleus3d = frustum_info['nucleus3d']
        frustum_stats_save_dir = os.path.join(stats_save_dir_per_frustum, f"{frustum_idx:03}")
        os.makedirs(frustum_stats_save_dir, exist_ok=True)
        slices_per_frustum = {cropped_image['z']: torch.tensor(cropped_image['mask']).float() for cropped_image in cropped_images if cropped_image['frustum_id'] == frustum_idx}
        slices_per_frustum = dict(sorted(slices_per_frustum.items())) # Sort the dictionary by the z coordinate
        for idx, (z, astro_image) in enumerate(slices_per_frustum.items()):
            save_path = os.path.join(frustum_stats_save_dir, f"{idx:03}.png")
            cell_surface = frustum_info['Cell Surface']
            primary_branches = frustum_info['Primary Branches']
            nuclei_surface = frustum_info['Nuclei Surface']
            number_nuclei = frustum_info['Number of Nuclei']
            number_astrocytes = frustum_info['Number of Astrocytes']
            text = f"Cell Surface: {cell_surface}\nPrimary Branches: {primary_branches}\nNuclei Surface: {nuclei_surface}\nNumber of Nuclei: {number_nuclei}\nNumber of Astrocytes: {number_astrocytes}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(astro_image, extent=[0.0, 1.0, 0.0, 1.0], aspect=1, origin='upper')
            ax.text(1.1, 0.5, text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            del fig, ax
    
    # Plot the overall statistics (collapsed masks)
    collapsed_astrocyte_mask = params['Collapsed Astrocyte Mask']
    collapsed_nuclei_mask = params['Collapsed Nuclei Mask']
    total_stats_save_path = os.path.join(stats_save_dir, "Total Statistics.png")
    text = f"Cell Surface: {total_cell_surface}\nPrimary Branches: {total_primary_branches}\nNuclei Surface: {total_nuclei_surface}\nNumber of Nuclei: {total_number_nuclei}\nNumber of Astrocytes: {total_number_astrocytes}"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(collapsed_astrocyte_mask, extent=[0.0, 0.5, 0.0, 0.5], aspect=1, origin='upper')
    ax.imshow(collapsed_nuclei_mask, extent=[0.0, 0.5, 0.5, 1.0], aspect=1, origin='upper')
    ax.text(1.1, 0.5, text, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([0, 1])
    fig.savefig(total_stats_save_path, dpi=300)
    fig.tight_layout()
    plt.close(fig)
    del fig, ax

def main(args):
    # Setup the configs
    cfg, d2cfg = setup(args)
    params = {
        'dir_counter': 1
    }

    # Setup the datasets
    print("=====Setting up the datasets=====")
    setup_datasets(cfg, d2cfg)

    # Run inference
    print("=====Running inference=====")
    params = run_inference(cfg, d2cfg, params)

    # Identify frustums
    print("=====Obtaining the frustums=====")
    params = get_frustums(cfg, params)

    # Run frustum segmentation
    print("=====Running frustum segmentation=====")
    params = run_frustum_segmentation(cfg, params)

    # Compute the statistics
    params = compute_stats(cfg, params)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.info(args)

    main(args)