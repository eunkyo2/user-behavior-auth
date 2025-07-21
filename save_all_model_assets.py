# save_all_model_assets.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

# --- 1. model/ 디렉터리 및 하위 디렉터리 생성 ---
os.makedirs("model/user_profiles", exist_ok=True)

# --- 2. (샘플) 데이터 생성 ---
n_samples, n_features = 1000, 8
X = np.random.normal(0, 1, (n_samples, n_features))
y = np.random.randint(0, 2, n_samples)

# --- 3. 모델 학습 & 저장 ---
clf = RandomForestClassifier()
clf.fit(X, y)
joblib.dump(clf, "model/model.joblib")
print("[✓] 모델 저장: model/model.joblib")

# --- 4. 스케일러 학습 & 저장 ---
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, "model/scaler.joblib")
print("[✓] 스케일러 저장: model/scaler.joblib")

# --- 5. 사용자별 프로파일 저장 (평균/표준편차) ---
def save_profile(user_id, data):
    profile = {
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist()
    }
    path = f"model/user_profiles/{user_id}.json"
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"[✓] {user_id} 프로파일 저장: {path}")

# --- user01 ~ user10까지 자동 생성 ---
for i in range(1, 11):
    # 각 사용자별로 평균/분산 다르게 하고 싶으면 아래처럼 조절 가능
    # 예시: user01~user10까지 평균 i, 분산 1
    user_data = np.random.normal(i, 1, (100, n_features))
    user_id = f"user{str(i).zfill(2)}"  # user01, user02, ..., user10
    save_profile(user_id, user_data)

print("\n[완료] model/ 안에 10명의 프로파일 포함 모든 파일 및 폴더가 자동으로 생성됨.")
