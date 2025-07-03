'''
Not suitable according to constems deployment format. NEED rechecking.
However main server code works

UPDATE -> This script works !!!!!
'''

import grpc
import numpy as np
from api import sam_tool_pb2
from api import sam_tool_pb2_grpc
import base64
import cv2
import ast
def main():
    # Connect to the server
    with grpc.insecure_channel('50051') as channel:
        stub = sam_tool_pb2_grpc.SamToolStub(channel)

        # Load an example image
        # image = Image.open("/home/cai00010/Pictures/sam2_localizer/PDI_24401_890_09122024144025_1_img.jpg")
        image_path = "/media/cai_002/New Volume2/sam2_localizer/PSI_20855_893_10022024110132_1_img.jpg"
        imgs = (cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        h,w=imgs.shape
        with open(image_path, 'rb') as fp:
            img = fp.read()
        image_data = base64.b64encode(img)

        # Send the image to the server
        request = sam_tool_pb2.features(image=image_data, height=h, width=w)
        response = stub.main(request)

        # Deserialize the numpy array
        arr = np.array(response.boxes, dtype=object)
        print(arr)
        fin = [ast.literal_eval(ar) for ar in arr]
        # array = array.reshape(response.shape)
        fin = np.array(fin)

        print("Received array with shape:", fin, type(fin))

if __name__ == "__main__":
    main()

