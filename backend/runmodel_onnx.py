import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt

classes = ['Bird', 'Flower', 'Hand', 'House', 'Mug', 'Pencil', 'Spoon', 'Sun', 'Tree', 'Umbrella']
classes.sort()

ort_session = ort.InferenceSession('savedmodel/model_final3.onnx')  # load the saved onnx model

# pre processing
def process(path):
    # pre process same as training
    image = Image.fromarray(plt.imread(path)[:, :, 3])  # read alpha channel
    image = image.resize((96, 96))
    image = (np.array(image) > 0.1).astype(np.float32)[None, :, :]

    return image[None]

# test the model
def test(path):
    image = process(path)
    output = ort_session.run(None, {'data': image})[0].argmax()

    print(classes[output], output)

    return classes[output]









