# Tensorflow Serving gRPC Client Python

To try it out, run MobileNetV1 locally with:

```
docker run -p 8501:8501 -p 8500:8500 \
  --mount type=bind,source=/Users/dnlserrano/Repos/mobilenet,target=/models/mobilenet \
  -e MODEL_NAME=mobilenet -t tensorflow/serving:latest
```

Then, in this folder run `poetry install` and start a virtualenv with `poetry shell`, followed by `python`. Paste the script or parts of it (`app.py`) into that session.
