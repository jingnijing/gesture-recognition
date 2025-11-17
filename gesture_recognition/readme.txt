#### train.ipynb
-> 제스처 인식 모델 학습 코드

#### inference.py 
-> 제스처 인식 추론용 코드 (windows 로컬 서버에서 flask로 열린 웹캠 url을 필요로 함 : windows 로컬에서 webcam_stream.py 파일 실행시킨 후 해당 파일 실행해야 함)

#### test.ipynb
-> test 데이터를 활용하여 confusion matrix 확인

#### test_data_collection.ipynb
-> test data 수집을 위한 코드 (windows 로컬 서버에서 flask로 열린 웹캠 url을 필요로 함 : windows 로컬에서 webcam_stream.py 파일 실행시킨 후 해당 파일 실행해야 함)

#### webcam_stream.py
-> 실시간 웹캠을 사용해야하는 inference.py 파일 등을 실행하기 전에 windows 로컬에서 해당 파일을 실행하여 flask로 웹캠 url을 열어줘야 함

#### requirements.txt
-> 라이브러리
(pip install -r requirements.txt)
