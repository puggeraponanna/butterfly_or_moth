#!env /bin/python

import onnxruntime
import numpy as np
from PIL import Image

from io import BytesIO
from starlette.applications import Starlette
from starlette.templating import Jinja2Templates
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse


templates = Jinja2Templates(directory='templates')
classes = ['butterfly', 'moth']
ort_session = onnxruntime.InferenceSession("models/butterfly_or_moth.onnx")
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def normalize(img):
    for i in range(3):
        img[i, :, :] = (img[i, :, :] - imagenet_stats[0][i]) / imagenet_stats[1][i]
    return img


def preprocess(img):
    img = img.resize((224, 224))
    img = np.asarray(img, np.float32).transpose(2, 0, 1)
    img = img / 256
    img = normalize(img)
    img = img.reshape(1, 3, 224, 224)
    return img


async def homepage(request):
    return templates.TemplateResponse('index.html', {'request': request})


async def upload(request):
    form_data = await request.form()
    byte_data = await (form_data["file"].read())
    image = preprocess(Image.open(BytesIO(byte_data)))
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outputs = ort_session.run(None, ort_inputs)
    output = np.argmax(ort_outputs[0])
    return JSONResponse({"prediction": classes[output]})


app = Starlette(debug=False, routes=[
    Route('/', homepage),
    Route('/upload', upload, methods=['POST']),
    Mount('/static', StaticFiles(directory='static'), name='static')
])
