import onnxruntime
import numpy as np
from PIL import Image
from pytorch_model import Classifier

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def test_model_onnx(onnx_model_path : str, image_paths : list, labels : list):

    assert len(image_paths) == len(labels)
    classifier = Classifier()

    session_model = onnxruntime.InferenceSession(onnx_model_path, None)

    for idx in range(len(image_paths)):

        label = labels[idx]
        image = Image.open(image_paths[idx])
        preprocessed_image = classifier.preprocess_numpy(image).unsqueeze(0)
        preprocessed_image = to_numpy(preprocessed_image)

        input_name = session_model.get_inputs()[0].name
        output_name = session_model.get_outputs()[0].name

        output = session_model.run([output_name], {input_name: preprocessed_image})
        assert np.argmax(output) == label

    
    print("Testing complete")

image_paths = ["n01440764_tench.jpeg", "n01667114_mud_turtle.JPEG"]
labels = [0, 35]

test_model_onnx(onnx_model_path="model.onnx", image_paths = image_paths, labels=labels)



