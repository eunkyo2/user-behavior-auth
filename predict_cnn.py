import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime

seq_len = 50
n_features = 8

# 1. 로그 함수 (파일 맨 앞에 선언)
os.makedirs("logs", exist_ok=True)
def log_event(message, log_file="logs/user_check.log"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {message}\n")

try:
    # 2. 모델 불러오기
    model = load_model('model/cnn_model.h5')
    log_event("모델 불러오기 성공")

    # 3. 실시간 입력 데이터 받기 (예: user05 스타일)
    user_idx = np.random.randint(1, 11)  # 1~10
    current_seq = np.random.normal(user_idx, 1, (1, seq_len, n_features))
    log_event(f"입력 시퀀스: 실제 유저 = user{str(user_idx).zfill(2)}")

    # 4. 예측
    pred = model.predict(current_seq)
    pred_user = np.argmax(pred)
    pred_prob = float(np.max(pred))
    user_name = f"user{str(pred_user+1).zfill(2)}"

    result_msg = f"AI 판별: {user_name}, 확률: {pred_prob:.4f}, 예측 분포: {np.round(pred, 3).tolist()}"
    print(result_msg)
    log_event(result_msg)

except Exception as e:
    log_event(f"에러 발생: {e}")
    print("예측 중 에러:", e)
