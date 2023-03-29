import onnxruntime
import numpy as np
from PIL import Image
from model import ONNXModel

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def test_model_onnx(onnx_model_path : str, image_paths : list, labels : list):

    assert len(image_paths) == len(labels)

    model = ONNXModel(onnx_model_path)

    for idx in range(len(image_paths)):

        label = labels[idx]
        prediction = model.predict(image_paths[idx])

        assert prediction == label
    
    print("Testing complete successfully")

def test_deployed_model(onnx_model, image_paths : list, labels : list):

    assert len(image_paths) == len(labels)

    for idx in range(len(image_paths)):

        label = labels[idx]
        prediction = onnx_model.predict(image_paths[idx])

        assert prediction == label
    
    print("Deployed model testing complete successfully")



