import grpc
import numpy as np
import tensorflow as tf

from PIL import Image

from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

img = Image.open('/Users/dnlserrano/Downloads/hopper.jpg')
img = img.resize((224, 224), Image.ANTIALIAS)
img = np.asarray(img)
img = img[np.newaxis, ...]
img = tf.keras.applications.mobilenet.preprocess_input(img)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'mobilenet'
request.inputs['input_1'].CopyFrom(tf.make_tensor_proto(img))

channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
response = stub.Predict(request, 5.0)

results = response.outputs['act_softmax'].float_val
results = np.array(results)

labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

print(imagenet_labels[results.argmax()+1])
