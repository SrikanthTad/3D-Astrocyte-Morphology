import os
import cv2
import pdb
import copy
import glob
import json
import torch
import shutil
import oiffile
import itertools
import numpy as np
import seaborn as sns
import pytorch_lightning as pl
import matplotlib.patches as patches
import detectron2.utils.comm as comm
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm
from queue import Queue
from sklearn.cluster import KMeans
from collections import OrderedDict
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

def register_dataset(data_path, image_format='.png'):
    data_path = copy.deepcopy(data_path)

    # Annotations and images directories/lists
    annotations_dir = os.path.join(data_path, "annotations")
    images_dir = os.path.join(data_path, "images")
    annotations_list = sorted(glob.glob(annotations_dir + "/*.json"))
    images_list = sorted(glob.glob(images_dir + "/*" + image_format))
    assert all([anns_path.split('/')[-1].split('.')[0] == img_path.split('/')[-1].split('.')[0] for anns_path, img_path in zip(annotations_list, images_list)])
    dataset_dicts = [None for _ in range(len(annotations_list))]

    # Load the dataset
    print(f"Loading the dataset from {data_path}")
    for idx, (anns_path, img_path) in enumerate(zip(annotations_list, images_list)):
        record = {}

        # Record preliminary information about the image
        file_name = img_path
        image_id = idx
        width, height = Image.open(img_path).size

        record["file_name"] = file_name
        record["image_id"] = image_id
        record["width"] = width
        record["height"] = height

        # Record the cells
        cells = []
        with open(anns_path, 'rb') as f:
            anns = json.load(f)
        for cell_info in anns['shapes']:
            points = cell_info['points']
            x1 = min(points[0][0], points[1][0]) # x min
            x2 = max(points[0][0], points[1][0]) # x max
            y1 = min(points[0][1], points[1][1]) # y min
            y2 = max(points[0][1], points[1][1]) # y max
            bbox = np.array([x1, y1, x2, y2])
            cell = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0
            }
            cells.append(cell)
        record["annotations"] = cells
        
        dataset_dicts[idx] = record
    
    return dataset_dicts

def setup_oib_dataset(cfg, oib_path, datasets_dir, image_format=".png", brightness_normalizer=4095.):
    """
    This function constructs a dataset and registers it with Detectron2 using an OIB file.

    Input:
        - cfg: Detectron2-format configs
        - oib_path: path to the OIB file
        - datasets_dir: directory where the datasets are stored
        - image_format: image format
        - brightness_normalizer: normalization constant for the 16-bit OIB data
    """
    def construct_empty_annotations(image_path, height, width):
        anns = {}
        anns['shapes'] = []
        anns['imagePath'] = "../" + image_path.split('/')[-1]
        anns['imageHeight'] = height
        anns['imageWidth'] = width
        return anns

    assert len(cfg.DATASETS.TEST) == 1, f"Expected only 1 test dataset. Got {len(cfg.DATASETS.TEST)} instead. Check cfg.DATASETS.TEST."
    save_dir = os.path.join(datasets_dir, cfg.DATASETS.TEST[0])
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    # Create sub-directories
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "annotations"), exist_ok=True)
    
    # Load the OIB file
    images_data = oiffile.imread(oib_path)
    # images_data = images_data[:, -2:, ...] # debug
    
    # Process the data
    images_data = torch.tensor(images_data[-1, ...] / brightness_normalizer) # The data is 16-bit, the last channel contains the astrocyte information
    images_data = images_data.unsqueeze(1)
    images_data = images_data.repeat(1, 3, 1, 1)
    images_data[:, 1, :, :] = 0 # Green channel is zero
    images_data[:, 2, :, :] = 0 # Blue channel is zero
    
    # Save the images
    data_dir = save_dir
    for idx, image in enumerate(images_data):
        image_save_path = os.path.join(data_dir, "images", str(idx) + image_format)
        anns_save_path = os.path.join(data_dir, "annotations", str(idx) + ".json")

        # Construct annotations
        anns = construct_empty_annotations(image_save_path, image.shape[0], image.shape[1])
        
        # Save the files
        save_image(image, image_save_path)
        with open(anns_save_path, 'w') as f:
            json.dump(anns, f)
    
    # Delete extra variables
    del images_data
    
    # Call the setup dataset function to proceed as usual
    setup_dataset(cfg, datasets_dir)

def setup_dataset(cfg, datasets_dir): 
    # Register train datasets
    for train_set_name in cfg.DATASETS.TRAIN:
        dataset_path = os.path.join(datasets_dir, train_set_name)
        
        # Register the dataset
        DatasetCatalog.register(train_set_name, lambda data_path=dataset_path : register_dataset(data_path))
        
        # Update the metadata
        MetadataCatalog.get(train_set_name).set(thing_classes=["astrocyte"], evaluator_type="custom-coco")
    
    # Register test datasets
    for test_set_name in cfg.DATASETS.TEST:
        dataset_path = os.path.join(datasets_dir, test_set_name)
        
        # Register the dataset
        DatasetCatalog.register(test_set_name, lambda data_path=dataset_path : register_dataset(data_path))
        
        # Update the metadata
        MetadataCatalog.get(test_set_name).set(thing_classes=["astrocyte"], evaluator_type="custom-coco")

class CustomCOCOEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, tasks=None, distributed=True, output_dir=None, *, max_dets_per_image=None, use_fast_impl=True, kpt_oks_sigmas=..., allow_cached_coco=True):
        super().__init__(dataset_name, tasks, distributed, output_dir, max_dets_per_image=max_dets_per_image, use_fast_impl=use_fast_impl, kpt_oks_sigmas=kpt_oks_sigmas, allow_cached_coco=allow_cached_coco)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)
    
def output2file(output, max_output, destination_file_path):
    """
    Save output to image
    """
    output = torch.tensor(output, dtype=torch.float32)
    output = output / float(max_output)
    save_image(output.permute(2, 0, 1), destination_file_path)

def save_cropped_astro_masks(cropped_images, save_dir, mask_normalizer=2.):
    """
    This function saves the given list of dicts into corresponding subfolders in save_dir,
    where each subfolder name corresponds to the frustum ID.

    Inputs:
        - cropped_images [List(Dict)]: list of dicts, where each dict contains the following keys: ['x1', 'x2', 'y1', 'y2', 'z', 'image', 'mask', 'frustum_id]
        - save_dir [str]: path to the save directory
    """
    for cropped_image in cropped_images:
        frustum_id = str(cropped_image['frustum_id'])
        z_value = str(cropped_image['z'])
        
        # Create directory if it doesn't exist
        frustum_folder = os.path.join(save_dir, frustum_id)
        os.makedirs(frustum_folder, exist_ok=True)
        
        # Save image
        image_path = os.path.join(frustum_folder, 'images', f'{z_value}.png')
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        # save_image(cropped_image['image'], image_path)
        cropped_image['image'].save(image_path)

        # Save mask
        mask_path = os.path.join(frustum_folder, 'masks', f'{z_value}.png')
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        save_image(torch.tensor(cropped_image['mask']).float().permute(2, 0, 1) / mask_normalizer, mask_path)
        # cropped_image['mask'].save(mask_path)

def construct_slices_tensor(slice_masks):
    """
    This function concatenates all masks per slice into a single 3D mask.
    """
    return torch.stack(list(slice_masks.values()))

# IoU calculation function. 
# Adopted from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] <= bb1['x2'], f"{bb1['x1']}, {bb1['x2']}"
    assert bb1['y1'] <= bb1['y2'], f"{bb1['y1']}, {bb1['y2']}"
    assert bb2['x1'] <= bb2['x2'], f"{bb2['x1']}, {bb2['x2']}"
    assert bb2['y1'] <= bb2['y2'], f"{bb2['y1']}, {bb2['y2']}"

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def spectral_clustering(adjacency_matrix, num_clusters):
    # Step 1: Create the Laplacian Matrix
    # Calculate the normalized Laplacian
    sqrt_inv_degree_matrix = np.diag(1 / np.sqrt(np.sum(adjacency_matrix, axis=1))) # D^(-1/2)
    normalized_laplacian = np.eye(len(adjacency_matrix)) - np.dot(sqrt_inv_degree_matrix, np.dot(adjacency_matrix, sqrt_inv_degree_matrix))
    
    # Step 2: Eigenvalue Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian)
    
    # Step 3: Select the Number of Clusters
    # You can use various methods to determine the number of clusters, such as the 'elbow' method.
    # For simplicity, we'll use a fixed number of clusters.
    
    # Step 4: Cluster the Data
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    cluster_assignments = kmeans.fit_predict(eigenvectors[:, 1:num_clusters + 1])

    # Calculate the average loss (inertia) of KMeans clustering
    average_loss = kmeans.inertia_
    
    return cluster_assignments, average_loss

# Function that retrieves all bounding boxes assigned to a particular cluster
def retrieve_cluster(bbox_list, idx):
    cluster = []
    for bbox_info in bbox_list:
        if bbox_info['cluster'] == idx:
            cluster.append(bbox_info)
    return cluster

# Function that converts a set of bounding boxes into 3D frustums. Return (xmin, xmax, ymin, ymax, zmin, zmax)
def bbox2frustum(cluster):
    # Extract bounding box coordinates (center, width, height)
    xmins = [min(bbox['x1'], bbox['x2']) for bbox in cluster]
    xmaxs = [max(bbox['x1'], bbox['x2']) for bbox in cluster]
    ymins = [min(bbox['y1'], bbox['y2']) for bbox in cluster]
    ymaxs = [max(bbox['y1'], bbox['y2']) for bbox in cluster]
    zs = [bbox['z'] for bbox in cluster]
    
    # Define the frustum
    x1 = min(xmins)
    y1 = min(ymins)
    x2 = max(xmaxs)
    y2 = max(ymaxs)
    z1 = min(zs)
    z2 = max(zs)

    frustum = {
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2,
        "z1": z1,
        "z2": z2,
    }

    return frustum

# Function that post-processes frustums
def postprocess_frustums(frustums_list):
    return frustums_list

def ensemble_prediction(image_batch, *models, device='cuda:0'):
    """
    This function is for doing prediction using ensemble models
    """
    with torch.no_grad():
        for model in models:
            model.eval()

        logits = [model(image_batch.to(device)) for model in models]
        logits = [torch.nn.functional.softmax(logit, dim=1).cpu() for logit in logits]
        logits = np.concatenate([logit[:, :, np.newaxis, :, :] for logit in logits], axis=2)

        logits = np.mean(logits, axis=2)

    return logits

def get_segmentation_nuclei(image, nuclei_models, preprocessing_fn):
    """
    Get the cropped of the cell, predict the mask and return the mask in the original shape
    """
    # we only need to get 1 channel for the nuclei
    image = np.array(image)[:, :, 0:1]
    image = np.repeat(image, 3, axis=-1)

    original_h, original_w, _ = image.shape

    image = cv2.resize(image, (192, 192))

    image = preprocessing_fn(image)

    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).float()

    with torch.no_grad():
        #output = nuclei_segmentation_model.model(image).numpy()
        output = ensemble_prediction(image, *nuclei_models)
        output = output[0].transpose(1, 2, 0).argmax(axis=-1, keepdims=True).astype(np.uint8)
        output = np.expand_dims(cv2.resize(output, (original_w, original_h)), axis=-1)

    return output

class SegmentationCell(pl.LightningModule):
    def __init__(self, architecture="unet", encoder_name="resnet34", N_INPUT_CHANNEL=3, NUCLEI_CLASSES=3):
        super().__init__()

        # TODO: add more architecture if you like
        if architecture == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=N_INPUT_CHANNEL,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=NUCLEI_CLASSES,                      # model output channels (number of classes in your dataset)
            )
        elif architecture == "unet++":
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=N_INPUT_CHANNEL,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=NUCLEI_CLASSES,                      # model output channels (number of classes in your dataset)
            )
        elif architecture == "manet":
            self.model = smp.MAnet(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=N_INPUT_CHANNEL,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=NUCLEI_CLASSES,                      # model output channels (number of classes in your dataset)
            )

    def forward(self, x):
        return self.model(x.float())
    
class NucleiTypes:
    BACKGROUND = 0
    NUCLEUS = 1
    BORDER = 2

def save_nuclei_from_oib(oib_path, save_dir, logger=None):
    """
    Take .oib files from source folder, extract images with nuclei and save it in destination folder
    """
    # Create directory if is doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    nuclei = oiffile.imread(oib_path)[0, ...]
    # nuclei = nuclei[-2:, ...] # debug
    n_nuclei = len(nuclei)

    # Save the nuclei images
    if logger is None:
        print(f"Saving {n_nuclei} images of nuclei slices")
    else:
        logger.info(f"Saving {n_nuclei} images of nuclei slices")
    for idx, image in tqdm(enumerate(nuclei), total=n_nuclei):
        image = np.expand_dims(image, axis=-1).astype(np.float32)
        image = np.repeat(image, 3, axis=-1)
        image[..., 0] = 0  # Set the red channel to 0
        image[..., 2] = 0  # Set the blue channel to 0
        
        # Save the image
        # save_path = os.path.join(destination_folder, f"{file_name}_{idx:03}.png")
        save_path = os.path.join(save_dir, f"{idx:03}.png")
        output2file(image, 4095., save_path) # 16-bit image, therefore 4095

def bfs3d(mask):
    def bfs(mask, start, visited, cluster_id):
        rows, cols, depth = mask.size()
        directions = [
            (1, 0, 0), 
            (-1, 0, 0), 
            (0, 1, 0), 
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1)
        ]  # Possible moves

        q = Queue()
        q.put(start)
        visited[start] = True

        while not q.empty():
            current_pixel = q.get()
            mask[current_pixel] = cluster_id

            for direction in directions:
                new_pixel = (current_pixel[0] + direction[0], current_pixel[1] + direction[1], current_pixel[2] + direction[2])

                if (
                    0 <= new_pixel[0] < rows
                    and 0 <= new_pixel[1] < cols
                    and 0 <= new_pixel[2] < depth
                    and mask[new_pixel] != 0.
                    and not visited[new_pixel]
                ):
                    q.put(new_pixel)
                    visited[new_pixel] = True
    
    # Perform BFS clustering in 3D
    binary_mask = mask.clone() # [slices, height, width]
    rows, cols, depth = binary_mask.size()
    visited = torch.zeros_like(binary_mask, dtype=torch.bool)
    cluster_id = 1  # Start cluster IDs from 1
    
    for i in range(rows):
        for j in range(cols):
            for k in range(depth):
                if binary_mask[i, j, k] != 0. and not visited[i, j, k]:
                    bfs(binary_mask, (i, j, k), visited, cluster_id)
                    cluster_id += 1

    return binary_mask

def bfs2d(mask):
    def bfs(mask, start, visited, cluster_id):
        rows, cols = mask.size()
        directions = [
            (1, 0), 
            (-1, 0), 
            (0, 1), 
            (0, -1),
        ]  # Possible moves

        q = Queue()
        q.put(start)
        visited[start] = True

        while not q.empty():
            current_pixel = q.get()
            mask[current_pixel] = cluster_id

            for direction in directions:
                new_pixel = (current_pixel[0] + direction[0], current_pixel[1] + direction[1])

                if (
                    0 <= new_pixel[0] < rows
                    and 0 <= new_pixel[1] < cols
                    and mask[new_pixel] == 1.
                    and not visited[new_pixel]
                ):
                    q.put(new_pixel)
                    visited[new_pixel] = True
    
    # Perform BFS clustering in 2D
    binary_mask = mask.clone()
    rows, cols = binary_mask.size()
    visited = torch.zeros_like(binary_mask, dtype=torch.bool)
    cluster_id = 1  # Start cluster IDs from 1
    
    for i in range(rows):
        for j in range(cols):
            if binary_mask[i, j] == 1. and not visited[i, j]:
                bfs(binary_mask, (i, j), visited, cluster_id)
                cluster_id += 1

    return binary_mask

def denoise_maskconn(mask3d, maskconn):
    assert mask3d.shape == maskconn.shape
    
    # Dictionary representing the indices corresponding to "Background", "Branch" and "Body"
    mc_classes = {"Background": [], "Astrocyte": []}
    recorded_idxs = []

    # Identify which category each cluster refers to
    for i in range(mask3d.shape[0]):
        for j in range(mask3d.shape[1]):
            for k in range(mask3d.shape[2]):
                mask3d_val = mask3d[i][j][k].item()
                mc_val = maskconn[i][j][k].item()
                if mc_val not in recorded_idxs:
                    if mask3d_val == 2. or mask3d_val == 1.:
                        mc_classes['Astrocyte'].append(mc_val)
                    else:
                        mc_classes['Background'].append(mc_val)
                    recorded_idxs.append(mc_val)
    
    # Replace the "Astrocyte" clusters which are not the biggest cluster
    class_dist = dict(zip(maskconn.unique(return_counts=True)[0].tolist(), maskconn.unique(return_counts=True)[1].tolist()))
    branch_dist = {branch_class: class_dist[branch_class] for branch_class in mc_classes['Astrocyte']}
    noise_mask = torch.ones(mask3d.shape)
    if len(branch_dist.values()) > 0:
        max_branch_idx = list(branch_dist.keys())[np.argmax(list(branch_dist.values()))]

        # Obtain the noise mask
        for i in range(maskconn.shape[0]):
            for j in range(maskconn.shape[1]):
                for k in range(maskconn.shape[2]):
                    if maskconn[i][j][k].item() in mc_classes['Astrocyte'] and maskconn[i][j][k] != max_branch_idx:
                        noise_mask[i][j][k] = 0.
    
    # Denoised mask
    denoised_mask = mask3d * noise_mask

    return denoised_mask, noise_mask

def find_closest_branch(mask, coords):
    coords = torch.tensor(coords).float()
    grid_x, grid_y = torch.meshgrid(torch.arange(mask.size(0)), torch.arange(mask.size(1)))
    grid = torch.cat((grid_x.unsqueeze(2), grid_y.unsqueeze(2)), dim=2)
    distance_map = torch.norm(grid - coords, p=2, dim=2)
    maxval = distance_map.max()
    distance_map[mask == 0.] = maxval
    min_coords = (distance_map==torch.min(distance_map)).nonzero()
    min_coord = min_coords[0] # Take the first minimum value
    branch_idx = mask[min_coord[0], min_coord[1]].item()
    return branch_idx

def plot_frustum_with_branches(branches3d, save_dir, frustum_idx):
    save_dir = os.path.join(save_dir, str(frustum_idx))
    os.makedirs(save_dir, exist_ok=True)
    n_colors = len(branches3d.unique())
    colors = torch.tensor(sns.color_palette('deep', n_colors=n_colors))
    if n_colors > 1:
        colors[1] = torch.tensor([0, 0, 0])
    for idx, slice in enumerate(branches3d):
        colored_slice = colors[(slice - branches3d.min()).int()]
        save_path = os.path.join(save_dir, f"{idx:03}.png")
        save_image(colored_slice.permute(2, 0, 1), save_path)

def compute_k_core(tensor, k):
    def get_degree_element(tensor, coords):
        x, y, z = coords
        degree = 0
        if tensor[x][y][z] == 0:
            return 0
        for i in range(max(0, x - 1), min(tensor.shape[0], x + 2)):
            for j in range(max(0, y - 1), min(tensor.shape[1], y + 2)):
                for k in range(max(0, z - 1), min(tensor.shape[0], z + 2)):
                    if (i, j, k) != (x, y, z):
                        degree += (tensor[i][j][k] != 0).int().item()
        return degree

    def get_degree_tensor(tensor):
        deg_tensor = torch.zeros(tensor.shape).int()
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                for k in range(tensor.shape[2]):
                    deg_tensor[i, j, k] = get_degree_element(tensor, (i, j, k))
        return deg_tensor
    
    kcore = tensor.clone()
    has_low_degree = True
    while has_low_degree:
        # Construct the degree tensor
        deg_tensor = get_degree_tensor(kcore)
    
        # Get the degree mask. The degree mask sets 0 for any element that has degree less than k
        deg_mask = ((kcore == 0) + (deg_tensor >= k)).float()

        # Check if there are low degree elements
        has_low_degree = 0 in deg_mask

        # Apply the mask
        kcore = kcore * deg_mask
    
    # Convert to the original type
    kcore = kcore.type(tensor.dtype)
    
    return kcore

def save_color_legend(save_dir, body_color, nuclei_color):
    overlap_color = 0.5 * (body_color + nuclei_color)
    overlap_color = overlap_color.tolist()
    body_color = body_color.tolist()
    nuclei_color = nuclei_color.tolist()
    save_path = os.path.join(save_dir, "colors_legend.png")

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create rectangles with RGB colors
    rect1 = patches.Rectangle((0.1, 0.8), 0.1, 0.1, edgecolor=body_color, facecolor=body_color, label='Body')
    rect2 = patches.Rectangle((0.1, 0.6), 0.1, 0.1, edgecolor=nuclei_color, facecolor=nuclei_color, label='Nuclei')
    rect3 = patches.Rectangle((0.1, 0.4), 0.1, 0.1, edgecolor=overlap_color, facecolor=overlap_color, label='Body & nuclei')

    # Add rectangles to the plot
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)

    # Manually create legend
    legend_elements = [rect1, rect2, rect3]
    legend_labels = ['Body', 'Nuclei', 'Body & Nuclei']
    ax.legend(legend_elements, legend_labels)

    # Add a text box in the middle
    text_box = plt.text(0.15, 0.3, "All other colors represent the branches", bbox=dict(facecolor='white', edgecolor='black'),
                        ha='center', va='center')

    # Set axis limits
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0.3, 1)

    # Hide axes for better appearance
    ax.axis('off')

    # Save the plot
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def save_total_colored_image(save_dir, image3d):
    # Save directory
    total_save_dir = os.path.join(save_dir, "total")
    os.makedirs(total_save_dir, exist_ok=True)

    # Save each slice
    for slice_idx, slice in enumerate(image3d):
        slice_save_path = os.path.join(total_save_dir, f"{slice_idx:03}.png")
        save_image(slice.permute(2, 0, 1), slice_save_path)