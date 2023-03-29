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

image_paths = ["images/n01440764_tench.jpeg", "images/n01667114_mud_turtle.JPEG"]
labels = [0, 35]

test_model_onnx(onnx_model_path="models/model.onnx", image_paths = image_paths, labels=labels)



