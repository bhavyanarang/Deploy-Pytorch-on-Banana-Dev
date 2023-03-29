import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

class ONNXModel:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_path):

        inputs = preprocessInput(input_path).getProcImage()
        outputs = self.session.run([self.output_name], {self.input_name: inputs})
        return np.argmax(outputs)

class preprocessInput:
    def __init__(self, image_path):

        self.image_path = image_path
        self.image = Image.open(image_path)
        self.preprocessed_image = self.preprocess(self.image)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def preprocess(self, img):
        resize = transforms.Resize((224, 224))   #must same as here
        crop = transforms.CenterCrop((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = resize(img)
        img = crop(img)
        img = to_tensor(img)
        img = normalize(img)
        img = self.to_numpy(img)

        return np.expand_dims(img, axis=0)

    def getProcImage(self):
        return self.preprocessed_image

model = ONNXModel("model.onnx")