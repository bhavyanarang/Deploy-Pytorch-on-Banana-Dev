import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class ONNXModel:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_path):

        images = Image.open(input_path)
        images = self.preprocess_input_images(images)

        outputs = self.session.run([self.output_name], {self.input_name: images})
        return np.argmax(outputs)

    def preprocess_input_images(self, img):
        resize = transforms.Resize((224, 224))   #must same as here
        crop = transforms.CenterCrop((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = resize(img)
        img = crop(img)
        img = to_tensor(img)
        img = normalize(img)
        img = to_numpy(img)

        return np.expand_dims(img, axis=0)

model = ONNXModel("model.onnx")