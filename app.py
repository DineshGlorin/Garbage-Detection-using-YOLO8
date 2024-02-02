from flask import Flask, render_template, Response
import cv2
import math
import webbrowser
from ultralytics import YOLO
model=YOLO("best.pt")
classNames=model.names
app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, img = camera.read()
        if not success:
            break
        else:
            results=model(img,conf=0.55)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(x1, y1, x2, y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = classNames[cls]
                    label = f'{class_name}{conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    print(t_size)
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(img, (x1, y1+25), c2, [0, 255, 0], -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 + 23), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=False)
