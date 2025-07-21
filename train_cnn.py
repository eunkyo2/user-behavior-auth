# predict_cnn.py
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime

seq_len = 50
n_features = 8

# 1. 로그 함수
os.makedirs("logs", exist_ok=True)
def log_event(message, log_file="logs/user_check2.log"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {message}\n")

# 2. 모델 불러오기
model = load_model('model/cnn_model.h5')

# 3. 실시간 입력 데이터 받기 (여기선 user03 스타일 가짜 데이터)
current_seq = np.random.normal(3, 1, (1, seq_len, n_features))  # (배치, 시퀀스길이, 피처수)

# 4. 예측
pred = model.predict(current_seq)
pred_user = np.argmax(pred)
pred_prob = float(np.max(pred))
user_name = f"user{str(pred_user+1).zfill(2)}"

# 5. 출력 & 로그 남기기
result_msg = f"AI가 판별한 유저: {user_name}, 확률: {pred_prob:.4f}, 분포: {np.round(pred, 3).tolist()}"
print(result_msg)
log_event(result_msg)
