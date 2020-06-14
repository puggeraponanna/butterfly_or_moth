#!env /bin/python

from io import BytesIO
from fastai.vision import open_image, Path, DataBunch, models, cnn_learner, load_learner
from starlette.applications import Starlette
from starlette.templating import Jinja2Templates
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.responses import JSONResponse


templates = Jinja2Templates(directory='templates')
path = Path('data')
learn = load_learner(path, "resn18.pkl")
classes = ['butterfly', 'moth']

async def homepage(request):
    return templates.TemplateResponse('index.html', {'request': request})


async def upload(request):
    form_data = await request.form()
    byte_data = await (form_data["file"].read())
    image = open_image(BytesIO(byte_data))
    image.save(Path("saved_image.jpg"))
    _, pred, losses = learn.predict(image)
    return JSONResponse({"prediction": classes[pred.item()]})


app = Starlette(debug=False, routes=[
    Route('/', homepage),
    Route('/upload', upload, methods=['POST']),
    Mount('/static', StaticFiles(directory='static'), name='static')
])
