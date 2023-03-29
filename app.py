from scripts.model import ONNXModel
from scripts.test_onnx import test_deployed_model

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    model = ONNXModel("models/model.onnx")


def inference(input_dict : dict) -> int:
    global model

    image_path = input_dict["image_path"]
    test_model = input_dict["test_model"]

    if test_model:
        image_paths = ["images/n01440764_tench.jpeg", "images/n01667114_mud_turtle.JPEG"]
        labels = [0, 35]
        test_deployed_model(model, image_paths = image_paths, labels=labels)

    result = model.predict(image_path)

    return result