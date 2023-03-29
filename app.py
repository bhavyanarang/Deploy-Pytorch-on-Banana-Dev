from scripts.model import ONNXModel

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    model = ONNXModel("modles/model.onnx")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(image_path : str) -> int:
    global model

    result = model.predict(image_path)

    # Return the results as a dictionary
    return result