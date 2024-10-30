import io
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = YOLO('yolo11x.pt')

@app.get('/')
async def name(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get('/upload_image')
async def name(request: Request):
    return templates.TemplateResponse('upload_image.html', {'request': request})


@app.get('/video_stream')
async def name(request: Request):
    return templates.TemplateResponse('video_stream.html', {'request': request})

@app.post('/upload_image')
async def upload_image(image: UploadFile = File(...)):
    contents = await image.read()
    np_image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    result_image = model(image)
    result_image = result_image[0].plot(show=True)
    succeeded, result_image_data = cv2.imencode('.jpeg', result_image)
    
    if not succeeded:
        return JSONResponse(status_code=500, content = {'error': 'Image processing failed'})
    
    return StreamingResponse(io.BytesIO(result_image_data.tobytes()), media_type="image/jpeg")

