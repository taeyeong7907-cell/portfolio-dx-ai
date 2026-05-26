# Project 02. 초전도체 데이터 기반 회귀·분류 예측 모델링 및 성능 비교

## 1. Project Overview

초전도체 데이터를 활용하여 임계온도 회귀 예측과 이진 분류 문제를 수행한 프로젝트입니다.

Baseline 모델과 개선 모델을 비교하며, RMSE, MAE, R², Accuracy, Precision, Recall, F1-score, ROC-AUC 등 다양한 평가 지표를 기준으로 모델 성능을 판단했습니다.

## 2. Problem Definition

- 초전도체의 임계온도를 예측하는 회귀 문제와 특정 기준에 따른 분류 문제를 함께 다뤘습니다.
- 단순히 복잡한 모델을 만드는 것이 아니라, 성능 지표를 기준으로 모델을 비교하고 적절한 모델을 선택하는 것이 핵심이었습니다.

## 3. Data / Modeling

| 항목 | 내용 |
|---|---|
| 입력 데이터 | 초전도체 특성 데이터 |
| 예측 문제 | 회귀, 분류 |
| 주요 모델 | DNN, Multi-Task Learning |
| 주요 기법 | StandardScaler, EarlyStopping, Dropout, BatchNormalization |
| 평가 지표 | RMSE, MAE, R², Accuracy, Precision, Recall, F1-score, ROC-AUC |

## 4. Experiment Summary

- Baseline 모델과 여러 개선 모델을 비교했습니다.
- 은닉층 수, 노드 수, Dropout, BatchNormalization, learning rate 등을 조정했습니다.
- 일부 개선 모델은 복잡도가 증가했지만 성능이 반드시 향상되지는 않았습니다.
- 최종 판단에서는 성능 수치와 학습 안정성을 함께 고려했습니다.

## 5. Key Results

| Task | Best Model | Main Metrics |
|---|---|---|
| Regression | Baseline | RMSE 10.7753, MAE 7.0533, R² 0.9015 |
| Classification | Baseline | Accuracy 0.9408, F1-score 0.9537, ROC-AUC 0.9808 |

- 개선 모델을 여러 차례 실험했지만, 최종 비교에서는 Baseline이 가장 안정적인 성능을 보였습니다.
- 이 프로젝트는 “복잡한 모델이 항상 좋은 결과를 내는 것은 아니다”라는 점을 확인한 모델 비교 경험입니다.

## 6. My Role

- 모델 성능 비교 결과 정리
- 실험 결과 표 및 그래프 구성
- 회귀·분류 평가 지표 해석
- 발표 자료 구성 보조

## 7. Visualization

성능표와 학습 곡선 이미지는 `images/` 폴더에 정리 예정입니다.

## 8. Limitations

- 교육 프로젝트 특성상 실제 산업 적용이나 서비스 배포는 수행하지 않았습니다.
- 비즈니스 문제보다는 AI 모델링 및 성능 평가 학습에 초점을 둔 프로젝트입니다.

## 9. Improvement Plan

- Feature importance 분석 추가
- 모델별 실험 조건표 정리
- 재현 가능한 실험 코드 정리
