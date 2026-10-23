import tensorrt as trt

engine_path = "/workspaces/isaac_ros-dev/src/model/bestperson.plan"
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    print(f"Tenseur : {name} | Forme : {shape}")
