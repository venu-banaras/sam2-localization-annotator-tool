
To build this repo's proto file use below command
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. api/sam_tool.proto


Takes image as input and returns boxes in numpy array format
