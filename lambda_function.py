import numpy as np

from io import BytesIO
from urllib import request
from PIL import Image

import tensorflow.lite as tflite
#import tflite_runtime.interpreter as tflite

#url = 'https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg'
image_size = (150, 150)
classes = [
    'dogs',
    'cats'
]

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

interpreter = tflite.Interpreter(model_path='dogs_cats_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=image_size)

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = preprocess_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)

    float_prediction = pred[0].tolist()

    return dict(zip(classes, float_prediction))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
