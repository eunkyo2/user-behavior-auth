import numpy as np
import json
import os
import random
from datetime import datetime

# 0. logs 디렉터리 및 로그 함수 준비
os.makedirs("logs", exist_ok=True)

def log_event(message, log_file="logs/user_check1.log"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {message}\n")

# 1. user01~user10 중 랜덤 프로파일 선택
user_id = f"user{str(random.randint(1, 10)).zfill(2)}"
profile_path = f"model/user_profiles/{user_id}.json"

with open(profile_path, "r") as f:
    profile = json.load(f)
mean = np.array(profile["mean"])
std = np.array(profile["std"])

print(f"선택된 유저 프로파일: {user_id}")

# 2. 실시간 입력 데이터 받았다고 가정 (지금은 랜덤 데이터)
current_data = np.random.normal(0, 1, mean.shape)  # 실제론 센서/터치 데이터

# 3. z-score 계산해서 “정상/비정상” 판별
z = np.abs((current_data - mean) / (std + 1e-6))  # 0으로 나누는 거 방지

# 4. 판별 & 로그 남기기
if (z > 3).any():
    print(f"{user_id} → 비정상 사용자! (ALERT)")
    log_event(f"{user_id} 판별 결과: 비정상 사용자! (ALERT)")
else:
    print(f"{user_id} → 정상 사용자!")
    log_event(f"{user_id} 판별 결과: 정상 사용자!")
