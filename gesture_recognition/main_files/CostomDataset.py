from torch.utils.data import Dataset,DataLoader,random_split
import torch
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import re
# 로그 파일로 출력 리다이렉션
#sys.stdout = open('custom_dataset_log.txt', 'w')


''' === 직원 데이터 (1프레임 = 1시퀀스 / Stop, None, Turn Clockwise, Turn Anticlockwise, Play 만) 를 포함한 전처리 코드 === '''

class CustomDataset(Dataset):
    def __init__(self, window_size, Folder_dir, stride=1):
        self.data_list = []
        self.window_size = window_size
        self.stride = stride
        self.log = []
        self.log_allstop = []

        # 라벨별 통계
        self.label_frame_counts = defaultdict(int)
        self.label_sequence_counts = defaultdict(int)

        all_files = sorted(os.listdir(Folder_dir))
        print(f"총 파일 수: {len(all_files)}\n")

        for filename in all_files:
            if not filename.endswith('.csv'):
                continue

            file_path = os.path.join(Folder_dir, filename)
            data = pd.read_csv(file_path, header=None).to_numpy()

            print(f"{filename} → 원본 shape: {data.shape}")

            features = torch.FloatTensor(data[:, 0:99])
            labels = torch.FloatTensor(data[:, 99])
            labels = labels.reshape(len(labels), -1)

            label_value = int(labels[0].item())
            self.label_frame_counts[label_value] += len(features)

            # ====== [1] 숫자_라벨명.csv or All_Stop 처리 / 단일 시퀀스 처리 ======
            if re.match(r'^\d+_[\w\s]+\.csv$', filename) or 'All_Stop' in filename:
                if len(features) >= window_size:
                    features_subset = features[:window_size]
                    label = labels[0]
                    self.data_list.append([features_subset, label])
                    self.label_sequence_counts[label_value] += 1
                    self.log_allstop.append(f"[{filename}] 단일시퀀스 처리 → 1시퀀스 추가 (shape: {features_subset.shape})")
                else:
                    self.log_allstop.append(f"[{filename}] 프레임 부족 → 건너뜀 (총 {len(features)}프레임)")

            # ====== [2] 일반 파일은 stride 기반 시퀀스 처리 / sliding window 방식 ======
            else:
                count = 0
                for i in range(0, len(features) - window_size + 1, self.stride):
                    features_subset = features[i:i + window_size]
                    label = labels[i]
                    self.data_list.append([features_subset, label])
                    count += 1
                    self.label_sequence_counts[int(label.item())] += 1
                self.log.append(f"[{filename}] 일반 처리 → {count} 시퀀스 생성")

        # 처리 로그 출력
        print("\n===== 단일 시퀀스 처리 로그 (숫자_라벨 또는 All_Stop) =====")
        for entry in self.log_allstop:
            print(entry)
        print(f"총 단일 시퀀스 파일 수: {len(self.log_allstop)}")

        print("\n===== 일반 처리 로그 (슬라이딩) =====")
        print(f"총 일반 제스처 파일 수: {len(self.log)} (상세 생략)")

        print("\n===== 라벨별 프레임/시퀀스 통계 =====")
        for label in sorted(self.label_frame_counts):
            print(f"  라벨 {label}: 총 {self.label_frame_counts[label]} 프레임 → {self.label_sequence_counts[label]} 시퀀스 생성")

        print(f"\n최종 생성된 시퀀스 수: {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        return x, y



''' === 제스처 10개 인식(none 제외) 시 사용했던 전처리 코드 (output model = model_gesture_10.pt) + 라벨 별 프레임 개수 및 시퀀스 개수 기록 기능 추가 === '''
# class CustomDataset(Dataset):
#     def __init__(self, window_size, Folder_dir, stride=1):
#         self.data_list = []
#         self.window_size = window_size
#         self.stride = stride
#         self.log = []
#         self.log_allstop = []

#         # 라벨별 프레임 수, 시퀀스 수 기록용 딕셔너리
#         self.label_frame_counts = defaultdict(int)
#         self.label_sequence_counts = defaultdict(int)

#         all_files = sorted(os.listdir(Folder_dir))
#         print(f"총 파일 수: {len(all_files)}\n")

#         for filename in all_files:
#             file_path = os.path.join(Folder_dir, filename)
#             data = pd.read_csv(file_path, header=None).to_numpy()
#             print(f"{filename} → 원본 shape: {data.shape}")

#             features = torch.FloatTensor(data[:, 0:99])
#             labels = torch.FloatTensor(data[:, 99])
#             labels = labels.reshape(len(labels), -1)

#             # 전체 프레임 수 기록 (첫 라벨 기준)
#             label_value = int(labels[0].item())
#             self.label_frame_counts[label_value] += len(features)

#             if 'All_Stop' in filename:
#                 if len(features) >= window_size:
#                     features_subset = features[:window_size]
#                     label = labels[0]
#                     self.data_list.append([features_subset, label])
#                     self.label_sequence_counts[label_value] += 1  # 시퀀스 수 기록
#                     self.log_allstop.append(f"[{filename}] All_Stop → 1시퀀스 추가 (shape: {features_subset.shape})")
#                 else:
#                     self.log_allstop.append(f"[{filename}] All_Stop → 프레임 부족 (총 {len(features)}프레임)")
#             else:
#                 count = 0
#                 for i in range(0, len(features) - window_size+1, self.stride):
#                     features_subset = features[i:i+window_size]
#                     label = labels[i]
#                     self.data_list.append([features_subset, label])
#                     count += 1
#                     self.label_sequence_counts[int(label.item())] += 1  # 시퀀스 수 기록
#                 self.log.append(f"[{filename}] 일반 제스처 → {count} 시퀀스 생성")

#         # 처리 로그 출력
#         print("===== All_Stop 처리 로그 =====")
#         for entry in self.log_allstop:
#             print(entry)
#         print(f"총 All_Stop 파일 수: {len(self.log_allstop)}")

#         print("\n===== 일반 제스처 처리 로그 (생략 가능) =====")
#         print(f"총 일반 제스처 파일 수: {len(self.log)} (상세 출력 생략)")

#         print("\n===== 라벨별 프레임/시퀀스 통계 =====")
#         for label in sorted(self.label_frame_counts):
#             print(f"  라벨 {label}: 총 {self.label_frame_counts[label]} 프레임 → {self.label_sequence_counts[label]} 시퀀스 생성")

#         print(f"\n총 생성된 시퀀스 수: {len(self.data_list)}")

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         x, y = self.data_list[idx]
#         x = torch.FloatTensor(x)
#         y = torch.FloatTensor(y)
#         return x, y

''' === 제스처 10개 인식(none 제외) 시 사용했던 전처리 코드 (output model = model_gesture_10.pt) === '''
# class CustomDataset(Dataset):
#     def __init__(self, window_size, Folder_dir):
#         self.data_list = []
#         self.window_size = window_size
#         self.log = []
#         self.log_allstop = []  # All_Stop 전용 로그 리스트

#         all_files = sorted(os.listdir(Folder_dir))
#         print(f"총 파일 수: {len(all_files)}\n")

#         for filename in all_files:
#             file_path = os.path.join(Folder_dir, filename)
#             data = pd.read_csv(file_path, header=None).to_numpy()
#             print(f"{filename} → 원본 shape: {data.shape}")

#             features = torch.FloatTensor(data[:, 0:99])
#             labels = torch.FloatTensor(data[:, 99])
#             labels = labels.reshape(len(labels), -1)

#             if 'All_Stop' in filename:
#                 if len(features) >= window_size:
#                     features_subset = features[:window_size]
#                     label = labels[0]
#                     self.data_list.append([features_subset, label])
#                     self.log_allstop.append(f"[{filename}] All_Stop → 1시퀀스 추가 (shape: {features_subset.shape})")
#                 else:
#                     self.log_allstop.append(f"[{filename}] All_Stop → 프레임 부족 (총 {len(features)}프레임)")
#             else:
#                 count = 0
#                 for i in range(len(features) - window_size):
#                     features_subset = features[i:i+window_size]
#                     label = labels[i]
#                     self.data_list.append([features_subset, label])
#                     count += 1
#                 self.log.append(f"[{filename}] 일반 제스처 → {count} 시퀀스 생성")

#         print("===== All_Stop 처리 로그 =====")
#         for entry in self.log_allstop:
#             print(entry)
#         print(f"총 All_Stop 파일 수: {len(self.log_allstop)}")

#         print("\n===== 일반 제스처 처리 로그 (생략 가능) =====")
#         print(f"총 일반 제스처 파일 수: {len(self.log)} (상세 출력 생략)")
        
#         print(f"\n총 생성된 시퀀스 수: {len(self.data_list)}")

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         x, y = self.data_list[idx]
        
#         x = torch.FloatTensor(x)
#         y = torch.FloatTensor(y)

#         return x, y
