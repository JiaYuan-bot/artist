# golang
#protoc  -I ./proto --go_out=plugins=grpc:./golang --go_opt=paths=source_relative proto/actor.proto proto/learner.proto

# python
# NEW, CORRECTED COMMAND
python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./python \
    --grpc_python_out=./python \
    ./proto/functions.proto
# mv python/idl/python/* ./python
# rm -rf python/idl
