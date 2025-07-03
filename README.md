# Objective
To use SAM2-Large as an annotator tool model for bounding boxes. 

### Working
This repository uses Docker and gRPC to create a dynamic and robust bounding box generator on any kind of images. This is however, specifically designed for SKU110K style dataset. Hence main target of this repository is to generate bounding boxes on SKUs as precisely as possible, allowing false positives to exist as a drawback, which would have to be addressed by human annotators.

### gRPC
To build this repo's proto file use below command
`python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. api/sam_tool.proto`

Main purpose of using gRPC is to allow this repo to be used as an AI agent for SKU localization. This takes image, height and width as input and returns bounding boxes in numpy array format. 
An image servicer microservice can read images from a mobile app or website and pass it to this AI agent, which will return bounding boxes. Now these bounding boxes can be passed to a database handler microservice that can dump these boxes into a NoSQL database like (MongoDB).

### SAM2 versions
This repo uses SAM2.1 large to return outputs. The hyperparameters used in this are pretty high and use lot of compute power and time to generate outputs, which is a design choice because of the density of products in SKU110K dataset. 
For 1 image it takes:-
1. 14.5Gb of GPU RAM (Tesla P100)
2. 16 second per image

However, I quantized this model to FP16 using TensorRT, and the new stats per image are:-
1. 12Gb of GPU RAM (Tesla P100)
2. 7 second per image

### Working
1. Create docker image using -> `sudo docker build -t image .`
2. Create docker container using -> `sudo docker run -it --name cont --shm-size=150G image`
3. Once the container is created the repo will start serving on port `50051`. 
4. Now this can be deployed as a microservice, where the consumer for this can be a mobile app which will first send it's captured image to a SQL/NoSQL DB. From  that DB another microservice can constantly check for new images and push them into a Kafka.
5. The Kafka will act as a data streamer for the consumer of this repo.
6. Now once this repo has generated its bounding boxes, it will send them to another microservice that works on database data dumping.