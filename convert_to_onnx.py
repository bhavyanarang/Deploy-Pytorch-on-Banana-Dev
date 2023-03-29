import torch
import torch.onnx
import onnx
from pytorch_model import Classifier, BasicBlock

def convert_model_to_onnx(model_path : str, output_path : str):

    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(model_path))

    model.eval()
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape)

    torch.onnx.export(model,
         dummy_input,
         output_path,
         export_params=True,
         opset_version=10,
         do_constant_folding=True,
         input_names = ['modelInput'],
         output_names = ['modelOutput'],
    ) 

    print("Conversion done")

convert_model_to_onnx("pytorch_model_weights.pth", "model.onnx")