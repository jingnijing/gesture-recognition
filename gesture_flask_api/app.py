from flask import Flask, render_template, Response
import cv2
import numpy as np
import torch
import time
from utils.preprocess import extract_hand_feature
from utils.model import CNN_LSTM

app = Flask(__name__)

# 모델 구조 정의
model = CNN_LSTM(input_size=99, output_size=64, units=32)
state_dict = torch.load('model/model_dict()_gesture_frame_30_indj.pt', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Softmax 함수 정의
softmax = torch.nn.Softmax(dim=1)

actions = ['Right', 'Left', 'Turn Clockwise', 'Turn Anticlockwise', 'Twinkle', 'Okay', 'Stop', 'Rock', 'Play','Slide', 'None']
seq = []
seq_length = 30
STABILITY_THRESHOLD = 3
COOLDOWN_SECONDS = 1
prev_label = None
stable_count = 0
this_action = '?'
last_action_time = 0 

# 웹캠
VIDEO_URL = 'http://192.168.0.103:5000/video'
cap = cv2.VideoCapture(VIDEO_URL)

def gen_frames():
    global seq, prev_label, stable_count, this_action

    while True:
        success, frame = cap.read()
        if not success:
            break

        img = cv2.flip(frame, 1)
        feature = extract_hand_feature(img)  # 프레임에서 feature 추출

        if feature is not None:
            seq.append(feature)

            if len(seq) >= seq_length:
                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                input_tensor = torch.FloatTensor(input_data)
                with torch.no_grad():
                    y_pred = model(input_tensor)
                    probs = softmax(y_pred)[0]
                    max_idx = torch.argmax(probs).item()
                    conf = probs[max_idx].item()
                    action = actions[max_idx]

                    if conf >= 0.2:
                        now = time.time()
                        
                        if (now - last_action_time) <= COOLDOWN_SECONDS:
                            pass
                        elif action == prev_label:
                            stable_count += 1
                            if stable_count >= STABILITY_THRESHOLD:
                                this_action = action
                                last_action_time = now
                                print(f"[예측] {action} ({conf:.2f})")
                        else:
                            stable_count = 1
                            prev_label = action
                    else:
                        stable_count = 0
                        prev_label = None

        # 제스처 오버레이
        cv2.putText(img, f'{this_action.upper()}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # 스트리밍용 인코딩
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
