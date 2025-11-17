#### 개발 목적
- 운전 중 운전자의 제스처에 따라 차량 인터페이스를 작동 및 중지 시키기 위한 제스처 인식 모델 개발
- 온디바이스 사용 가능한 경량화된 모델로 개발
- 총 11개의 제스처
  - Right (0) : 다음 트랙으로 이동
  - Left (1) : 이전 트랙으로 이동
  - Turn Clockwise (2) : 볼륨 증가
  - Turn AntiClockwise (3) : 볼륨 감소
  - Twinkle (4) : AI 기능 호출
  - Okay (5) : '예' 또는 '선택'
  - Stop (6) : 음악 일시 정지, 중지
  - Rock (7) : 음악 추천 기능 호출
  - Play (8) : 음악 재생
  - Slide (9) : 진행 중인 AI 전 기능 종료
  - None (10) : 위 10가지에 속하지 않는 제스처들

#### [gesture_dependencies]
제스처 모델 개발 requirements.txt 초기 version

#### [gesture_flask_api]
제스처 모델 개발 후 flask로 배포

#### [gesture_recognition]
제스처 모델 정의/학습/추론 코드 및 학습 데이터 폴더
(비교적 최신 requirements 저장되어 있음) 
모델 추가 학습 및 재학습 시 해당 파일, 환경 세팅해서 이어서 개발 가능


