import io
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import json
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from camera import detect_from_camera

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = YOLO('yolo11x.pt')

#get user back to home page
@app.get('/')
async def name(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get('/video_stream_dark')
async def name(request: Request):
    return templates.TemplateResponse("video_stream_dark.html", {"request": request})


#get user to the upload_image.html
@app.get('/upload_image')
async def name(request: Request):
    return templates.TemplateResponse('upload_image.html', {'request': request})

#get user to the video_stream.html
@app.get('/video_stream')
async def name(request: Request):
    return templates.TemplateResponse('video_stream.html', {'request': request})

#post an image from the upload_image screen to the backend, returns the image back to frontend
@app.post('/upload_image')
async def upload_image(image: UploadFile = File(...)):
    contents = await image.read()
    np_image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    image = model(image)
    
    #takes bounding box info to make it readable
    detections = []
    for box in image[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        label = model.names[int(box.cls[0])]
        detections.append({
            "label": label,
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2]
        })
    #converts the info to metadata and stores into a file --> accessed from '/image_data' endpoint
    with open("image_data.json", "w",) as file:
        json.dump(detections, file)
    
    
    #draws the bounding boxes, onto the image
    result_image = image[0].plot(show=False)

    #resizes the image if the height is more than max pixels (500)
    image_y, image_x = result_image.shape[:2]
    if image_y > 500:
        new_x, new_y = resize_image(image_x, image_y)
        result_image = cv2.resize(result_image, (new_x, new_y))
    
    succeeded, result_image_data = cv2.imencode('.jpeg', result_image)
    
    if not succeeded:
        return JSONResponse(status_code=500, content = {'error': 'Image processing failed'})
    
    return StreamingResponse(io.BytesIO(result_image_data.tobytes()), media_type="image/jpeg")

def resize_image(x, y):
    max_y = 500
    new_x = (max_y * x) // y
    return new_x, max_y

#opens the video streaming feed popup for users
@app.get('/video_stream_feed')
async def video_stream_function(request: Request):
    return StreamingResponse(detect_from_camera(request), media_type="multipart/x-mixed-replace; boundary=frame")

#sends out bounding box data file to the user
@app.get('/image_data')
async def get_image_data():
    return FileResponse("image_data.json", media_type="application/json", filename="image_data.json")

