#### requirements_init.txt
-> 필수로 존재해야 하는 라이브러리 4개
-> torch 2.4.0 로 명시해서 requirements_init.txt 로 설치하면 자동으로 CUDA, nvidia 관련 패키지가 12.x 버전으로 설치됨
-> 만약 실제 개발환경처럼 cu11.8 버전으로 설치해야 한다면 requirements_init.txt install이 아닌

1) pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118
2) pip install mediapipe==0.10.21
install 해주면 됨

#### requirements_ori.txt
-> 위 pip install 코드 2개 돌려서 설치된 모든 패키지 (cu118)

#### requirements_min.txt
-> requirements_init.txt install로 설치된 모든 패키지 (cu12.x)
