import mediapipe as mp
import numpy as np
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_hand_feature(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None

    for res in result.multi_hand_landmarks:
        joint = np.zeros((21, 4))
        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
        
        # 정규화 (단위 벡터로 만듬 / 벡터의 길이를 1로 만듬)
        v = v2 - v1
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
        
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]))
        angle = np.degrees(angle)
        d = np.concatenate([joint.flatten(), angle])
        return d.astype(np.float32)

    return None
