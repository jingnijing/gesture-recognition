from flask import Flask, Response, render_template_string
import cv2

app = Flask(__name__)

def generate_frames():
    # 웹캠 설정
    cap = cv2.VideoCapture(0)
    print('웹캠 연결 완료')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # 좌우반전 제거
            frame = cv2.flip(frame, 1)
            # 프레임을 JPEG 형식으로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # 멀티파트 메시지 형식으로 프레임 생성
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # HTML 페이지에서 비디오 스트리밍을 보여주는 `<img>` 태그 포함
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <title>Video Streaming</title>
    </head>
    <body>
    <h1>Video Streaming</h1>
    <img src="/video" width="640" height="480">
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video')
def video():
    # 비디오 스트리밍을 위한 Response 객체 생성
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
