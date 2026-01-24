
```
pip install grpcio-tools
pip install mypy-protobuf

mkdir -p generate
python -m grpc_tools.protoc     -I.  --mypy_out=generate/   --python_out=generate/      */*.proto