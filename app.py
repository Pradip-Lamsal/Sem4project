from flask import Flask, render_template, Response, jsonify
import cv2
import torch
from ultralytics import YOLO
import random

app = Flask(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt').to(device)

# Price and recommendation mapping
price_map = {
    'person': 10000,
    # Add more classes and their prices here
}

recommendation_map = {
    'person': ['person A', 'person B'],
    # Add more classes and their recommendations here
}

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            resized_frame = cv2.resize(frame, (320, 240))
            results = model.predict(source=resized_frame, device=device, conf=0.25)
            detected_object = None
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0] * 2)
                    confidence = box.conf[0]
                    cls = int(box.cls[0])
                    label = f"{model.names[cls]} {confidence:.2f}"
                    detected_object = model.names[cls]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_price')
def get_price():
    detected_object = random.choice(list(price_map.keys()))  # Simulate detection
    price = price_map.get(detected_object, 'N/A')
    recommendations = recommendation_map.get(detected_object, [])
    return jsonify({'object': detected_object, 'price': price, 'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
