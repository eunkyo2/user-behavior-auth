# train_cnn.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import glob

# 설정
n_users = 10
seq_len = 50       # 시퀀스 길이
n_features = 8     # 피처 개수(데이터에 따라 맞춰야 함)
samples_per_user = 200

# 데이터 로딩 및 시퀀스화 함수
def make_sequences(data, seq_len):
    # data: (총 row, n_features)
    seqs = []
    for i in range(len(data) - seq_len + 1):
        seqs.append(data[i:i+seq_len])
    return np.stack(seqs)

X = []
y = []

# 1. data 폴더에서 user별로 데이터 읽어서 시퀀스화
for i in range(1, n_users+1):
    user_id = f"user{str(i).zfill(2)}"
    csv_path = f"data/{user_id}.csv"
    if not os.path.exists(csv_path):
        # 샘플 데이터 없으면 가짜 데이터 생성
        user_data = np.random.normal(i, 1, (samples_per_user+seq_len-1, n_features))
        pd.DataFrame(user_data).to_csv(csv_path, index=False, header=False)
    else:
        user_data = pd.read_csv(csv_path, header=None).values
    user_seqs = make_sequences(user_data, seq_len)  # (샘플수, seq_len, n_features)
    X.append(user_seqs)
    y += [i-1]*len(user_seqs)
X = np.vstack(X)
y = to_categorical(y, num_classes=n_users)

# 2. 학습/검증 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. 1D-CNN 모델 설계
model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=(seq_len, n_features)),
    Dropout(0.2),
    Conv1D(64, 3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(n_users, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 학습
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 5. 평가 및 저장
loss, acc = model.evaluate(X_test, y_test)
print(f"테스트 정확도: {acc:.4f}")

os.makedirs("model", exist_ok=True)
model.save("model/cnn_model.h5")
print("[✓] 딥러닝 모델 저장: model/cnn_model.h5")
