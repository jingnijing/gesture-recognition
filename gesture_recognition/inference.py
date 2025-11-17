import cv2
import mediapipe as mp
import numpy as np
import torch
import time

# 모델 로드
model = torch.load('./main_models_add/model_gesture_frame_15_indj_2.pt', map_location='cpu')
model.eval()

# 제스처 라벨
actions = ['Right', 'Left', 'Turn Clockwise', 'Turn Anticlockwise', 'Twinkle', 'Okay', 'Stop', 'Rock', 'Play','Slide', 'None']
# 모델 입력 시퀀스의 길이(프레임 수)
seq_length = 30
# 예측 결과가 연속으로 몇 번 이상 반복되어야 확정 판단할지 설정하는 값
STABILITY_THRESHOLD = 3
# 하나의 제스처가 확정된 후 다시 인식하기까지의 쿨타임
COOLDOWN_SECONDS = 1

# 안정화 변수
prev_label = None
stable_count = 0
this_action = '?'
last_action_time = 0
latency_logged = False


# MediaPipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 웹캠 입력
VIDEO_URL = 'http://192.168.0.103:5000/video'
cap = cv2.VideoCapture(VIDEO_URL)

# 입력 시퀀스 저장용
seq = []

# 소프트맥스
softmax = torch.nn.Softmax(dim=1)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            # 랜드마크 및 각도 추출
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            # 프레임이 하나 들어올 때마다 seq 리스트에 한 프레임 분량의 데이터를 1개씩 추가
            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            
            # seq의 길이가 seq_length보다 짧으면 예측을 하지 않음
            if len(seq) < seq_length:
                continue
            
            # seq가 30개가 되면 모델 입력 시작 / 항상 마지막 30개(seq[-30:])를 가져와서 모델에 넣고 있음 = 1 프레임씩 밀리며 예측 = 슬라이딩 윈도우 방식의 예측
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            input_tensor = torch.FloatTensor(input_data)
            
            # 예측 지연 시간 측정 시작
            start_time = time.perf_counter()   
            
            # 예측
            y_pred = model(input_tensor)
            probs = softmax(y_pred)[0]
            
            # 예측 지연 시간 측정 종료
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            if not latency_logged:
                print(f'예측 지연 시간 (1회 측정): {latency_ms:.3f} ms')
                latency_logged = True
                        
            max_idx = torch.argmax(probs).item()
            conf = probs[max_idx].item()
            action = actions[max_idx]

            # 확률 표시
            for i, prob in enumerate(probs):
                text = f'{actions[i]}: {prob:.2f}'
                cv2.putText(img, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 현재 none 포함 11개의 제스처 인식 -> 0.21 값이 가장 높은 예측률
            if conf >= 0.20:
                now = time.time()

                if (now - last_action_time) <= COOLDOWN_SECONDS:
                    pass  # 쿨타임 중이면 아무것도 하지 않음
                elif action == prev_label:
                    stable_count += 1
                    if stable_count >= STABILITY_THRESHOLD:
                        this_action = action
                        last_action_time = now
                        print(f"제스처 인식 : {this_action}")
                else:
                    stable_count = 1
                    prev_label = action
            else:
                stable_count = 0
                prev_label = None

            # 손목 위치에 제스처 이름 표시
            wrist_x = int(res.landmark[0].x * img.shape[1])
            wrist_y = int(res.landmark[0].y * img.shape[0]) + 20
            cv2.putText(img, f'{this_action.upper()}',
                        org=(wrist_x, wrist_y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.2, color=(255, 255, 255), thickness=2)
    else:
        # 손 없을 땐 왼쪽 상단에 제스처 표시
        cv2.putText(img, f'{this_action.upper()}',
                    org=(30, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.2, color=(200, 200, 200), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
