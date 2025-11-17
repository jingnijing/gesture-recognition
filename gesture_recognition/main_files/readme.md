#### model.py
- cnn lstm 모델 정의
#### make_dataset.ipynb 
- 손 제스처 랜드마크 데이터셋(csv) 생성을 위한 코드
#### CustomDataset.py
- 학습 전, 데이터 전처리 시 사용되는 코드 / 15 혹은 30 프레임 당 시퀀스(슬라이딩 윈도우, 단일 시퀀스) 처리 등
#### data_clone.ipynb
- 생성된 데이터셋(csv) 을 기반으로 데이터 복제/증강


#### (참고)
- model.py로 모델 파라미터 및 레이어 수정 가능
- make_data.ipynb로 손 랜드마크를 추출
- CustomDataset.py로 데이터 전처리 (필요에 따라)
- data_clone.ipynb로 데이터셋 복제 및 증강 가능 (필요에 따라)


