import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import pandas as pd
from PIL import Image
import numpy as np
import torch
import sys
sys.path.append('api/')
# import grpc
from concurrent import futures
from api import sam_tool_pb2
from api import sam_tool_pb2_grpc
import base64
import io
# import ast
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SamToolServicer(sam_tool_pb2_grpc.SamToolServicer):

    def __init__(self, device):
        
        self.sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.sam2 = build_sam2(self.model_cfg, self.sam2_checkpoint, device=device, apply_postprocessing=False)


        self.mask_generator_6 = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=32,
            points_per_batch=128,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.88,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.3,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=7000,
            use_m2m=True,
        )

    def iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Parameters:
            box1 (tuple): (xmin, ymin, xmax, ymax) for the first box.
            box2 (tuple): (xmin, ymin, xmax, ymax) for the second box.

        Returns:
            float: IoU value.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0

    def nms_from_csv(self, df, threshold):
        """
        Apply Non-Maximum Suppression (NMS) on bounding boxes.

        Parameters:
            csv_path (str): Path to the CSV file containing bounding boxes.
            threshold (float): IoU threshold for NMS.

        Returns:
            DataFrame: Filtered bounding boxes after applying NMS.
        """


        required_columns = ['xmin', 'ymin', 'xmax', 'ymax']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        filtered_boxes = []

        while boxes:
            chosen_box = boxes.pop(0)
            filtered_boxes.append(chosen_box)
            boxes = [box for box in boxes if self.iou(chosen_box, box) < threshold]

        filtered_df = pd.DataFrame(filtered_boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])
        return filtered_df

    def remove_invalid_boxes(self, df, image_path, threshold):
        """
        Remove bounding boxes where dimensions are below a percentage of image dimensions.

        Parameters:
            df (DataFrame): DataFrame containing bounding boxes.
            image (str): Path to the folder containing images.
            threshold (float): Minimum ratio of box size to image size.

        Returns:
            DataFrame: Filtered bounding boxes.
        """
        filtered_boxes = []

        # image_path = Image.open(image)
        width, height = image_path.size

        for _, row in df.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

            box_width = xmax - xmin
            box_height = ymax - ymin

            if box_width < threshold * width and box_height < threshold * height:
                filtered_boxes.append([xmin, ymin, xmax, ymax])

        filtered_df = pd.DataFrame(filtered_boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])
        return filtered_df

    def remove_small_area_boxes(self, df, area_threshold):
        """
        Remove bounding boxes with an area significantly smaller than the average.

        Parameters:
            df (DataFrame): DataFrame containing bounding boxes.
            area_threshold (float): Minimum ratio of box area to the average area.

        Returns:
            DataFrame: Filtered bounding boxes.
        """
        df['area'] = (df['xmax'] - df['xmin']) * (df['ymax'] - df['ymin'])
        avg_area = df['area'].mean()
        filtered_df = df[df['area'] >= area_threshold * avg_area].drop(columns=['area'])
        return filtered_df

    def remove_large_boxes(self, df, image_path, size_threshold):
        """
        Remove bounding boxes that are too large compared to image dimensions.

        Parameters:
            df (DataFrame): DataFrame containing bounding boxes.
            image (str): Path to the folder containing the images.
            size_threshold (float): Maximum ratio of box size to image size.

        Returns:
            DataFrame: Filtered bounding boxes.
        """
        filtered_boxes = []
        # image_path = Image.open(image)
        width, height = image_path.size

        for _, row in df.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

            box_width = xmax - xmin
            box_height = ymax - ymin

            if box_width <= size_threshold * width and box_height <= size_threshold * height:
                filtered_boxes.append([xmin, ymin, xmax, ymax])

        filtered_df = pd.DataFrame(filtered_boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])
        return filtered_df

    def remove_edge_boxes(self, df, image_path, edge_threshold):
        """
        Remove bounding boxes near image edges and return the result as a NumPy array.

        Parameters:
            df (DataFrame): DataFrame containing bounding boxes.
            image (str): Path to the folder containing the images.
            edge_threshold (int): Minimum distance from the edges.

        Returns:
            numpy.ndarray: Filtered bounding boxes as a NumPy array.
        """
        # image_path = Image.open(image)
        width, height = image_path.size
        filtered_boxes = []

        for _, row in df.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

            if (xmin > edge_threshold and ymin > edge_threshold and
                    xmax < width - edge_threshold and ymax < height - edge_threshold):
                filtered_boxes.append((xmin, ymin, xmax, ymax))

        return np.array(filtered_boxes)





    '''Generate bounding boxes as a DataFrame using SAM2'''
    def generate_bounding_boxes(self, image_dir):
        bbox_data = []

        
        # image = Image.open(image_dir) 
        image = image_dir.convert("RGB") 
        image_np = np.array(image)
        masks = self.mask_generator_6.generate(image_np)

        print(f"Total masks: {len(masks)}\n")

        for mask in masks:
            x_min, y_min, width, height = mask['bbox']
            x_max = x_min + width
            y_max = y_min + height
            bbox_data.append([x_min, y_min, x_max, y_max])

        df = pd.DataFrame(bbox_data, columns=["xmin", "ymin", "xmax", "ymax"])
        
        return df

    '''Apply box suppression directly on the DataFrame'''
    def apply_box_suppression(self, bbox_df, image):
        filtered_df = self.nms_from_csv(bbox_df, 0.2)
        print(f"After NMS: {filtered_df.shape[0]} boxes")
        
        filtered_df = self.remove_invalid_boxes(filtered_df, image, 0.8)
        print(f"After invalid box removal: {filtered_df.shape[0]} boxes")

        filtered_df = self.remove_small_area_boxes(filtered_df, 0.28)
        print(f"After small area removal: {filtered_df.shape[0]} boxes")
        
        filtered_df = self.remove_large_boxes(filtered_df, image, 0.25)
        print(f"After large box removal: {filtered_df.shape[0]} boxes")
        
        n_arr = self.remove_edge_boxes(filtered_df, image, 4)
        print(f"After edge box removal: {filtered_df.shape[0]} boxes")

        return n_arr

    def main(self, request, context):
        print('got request')
        # data = request.image
        # h = request.height
        # w = request.width
        img = "/media/cai_002/New Volume2/sam2_localizer/PSI_20855_893_10022024110132_1_img.jpg"
        with open(img, 'rb') as fp:
            imgs = fp.read()
        img_comp = base64.b64encode(imgs)

        image_arr = base64.b64decode(img_comp)
        image_arr = Image.open(io.BytesIO(image_arr))
        bbox_df = self.generate_bounding_boxes(image_arr)

        # print(image_arr, type(image_arr))
        final_df = self.apply_box_suppression(bbox_df, image_arr)
        print(final_df.shape)
        # a =[[1,2,3,4],[5,6,7,8],[9,0,9,8],[7,6,5,44]]
        # a = np.array(a)
        final_arr = list(map(str, final_df.tolist()))
        # print(final_df, type(final_df))
        print("Sending response", final_arr)


        # return sam_tool_pb2.results(
        #       boxes= final_arr
        # )


def serve(device):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sam_tool_pb2_grpc.add_SamToolServicer_to_server(SamToolServicer(device=device), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:6")
    else:
        # device = torch.device('cpu')
        print("GPU not available, exiting ...")
        exit(0)

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    np.random.seed(3)
    print("Started SAM2 service")
    # serve(device)
    obj = SamToolServicer(device=device)
    obj.main(device, 12)