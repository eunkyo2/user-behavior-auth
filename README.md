## AI 기반 사용자 인증/식별 프로젝트

## 프로젝트 개요

이 프로젝트는 **안드로이드 단말의 터치·센서 데이터**를 활용하여  
- 등록된 사용자인지(정상/비정상) 판별  
- 또는 10명의 사용자 중 "누구"인지 AI가 예측  
하는 시스템을 개발하는 것이 목적입니다.

머신러닝(RandomForest)과 딥러닝(시계열 1D-CNN)을 모두 적용하여  
실제 사용자 인증, 무단 접근 탐지, 자동 사용자 식별 등 다양한 환경에 사용할 수 있습니다.

---

## 폴더/파일 구조

ai-auth-project/
├── save_all_model_assets.py # RandomForest/프로필 생성 및 저장
├── check_user_profile.py # 평균/분산 기반 정상/비정상 판별
├── train_cnn.py # 시계열 CNN 딥러닝 학습
├── predict_cnn.py # CNN 기반 유저 판별
├── model/
│ ├── model.joblib # RandomForest 저장
│ ├── cnn_model.h5 # 딥러닝 모델 저장
│ └── user_profiles/ # 사용자별 평균/분산 json
├── data/
│ └── user01.csv # (예시) 사용자별 행동데이터 (미완)
├── logs/
│ ├── user_check1.log # 정상/비정상 판별 로그
│ └── user_check2.log # 유저 식별(AI) 판별 로그
└── README.md


---

## 주요 기능 및 코드 설명

- **정상/비정상 판별:**  
  - `save_all_model_assets.py`로 생성된 user 프로필(평균/표준편차) 기반  
  - `check_user_profile.py`로 실시간 데이터와 비교,  
    결과는 `logs/user_check1.log`에 저장됨

- **사용자 식별(1~10번):**  
  - `train_cnn.py`로 시계열 1D-CNN 모델 학습/저장  
  - `predict_cnn.py`에서 실시간 행동 시퀀스 입력 시  
    10명 중 가장 유사한 user를 예측,  
    결과는 `logs/user_check2.log`에 저장됨

---

## 실행 방법

1. **필요 패키지 설치**
    ```
    pip install numpy pandas scikit-learn tensorflow
    ```

2. **정상/비정상 판별 실험**
    - (1) user_profiles 폴더 자동 생성  
    - (2) 실시간 데이터 판별
    ```
    python save_all_model_assets.py
    python check_user_profile.py
    ```
    - 결과는 `logs/user_check1.log`에서 확인

3. **CNN 기반 사용자 식별 실험**
    - (1) 딥러닝 모델 학습 (가짜 데이터 or 실제 데이터 사용)
    - (2) 실시간 예측
    ```
    python train_cnn.py
    python predict_cnn.py
    ```
    - 결과는 `logs/user_check2.log`에서 확인

---

## 데이터 구조

- **user_profiles/\*** : 각 사용자 평균/분산(json)  
- **data/userXX.csv** : 한 줄마다 `[x좌표, y좌표, 터치시간, acc_x, ...]` 식의 8개 피처  
- **logs/\*** : 주요 판별 로그 자동 기록

---

## 작업 및 개발 방향성

- **정상/비정상 판별**은 규칙 기반(평균/분산),  
  **사용자 식별**은 딥러닝(CNN) 기반 AI로 확장 적용
- **실제 서비스**를 위해선  
  - 실제 행동 데이터 수집 → data/에 저장  
  - 학습/추론 자동화  
  - 로그 기록으로 결과 추적 및 성능 분석
- **협업/공유**는 깃허브로, 코드·설명·예시 데이터 중심 관리

---

## 문의/참고

- 담당자: 허은교
- 참고자료: 

