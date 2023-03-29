import banana_dev as banana
import os
import time

banana_api_key = os.environ.get('BANANA_API_KEY')
model_key = os.environ.get("YOUR_MODEL_KEY")

def test_deployed_model(input_path, print_time = True):
    start_time = time.time()
    out = banana.run(banana_api_key, model_key, input_path)
    end_time = time.time()

    print("Predicted class id : ", out)

    if print_time:
        print("Time taken to call model : {} seconds".format(end_time - start_time))


model_input_path = "images/n01440764_tench.jpeg"

test_deployed_model(model_input_path)