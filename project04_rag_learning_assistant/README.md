# Project 04. RAG 기반 AIVLE School 학습도우미 구축 및 사용자 질문 흐름 개선

## 1. Project Overview

AIVLE School 관련 문서를 기반으로 사용자의 질문에 답변하는 RAG 기반 학습도우미 프로젝트입니다.

단순 챗봇이 아니라 문서 기반 검색과 생성형 AI를 결합하여 답변 신뢰도를 높이고, 사용자가 학습 흐름을 이어갈 수 있도록 예상 질문과 꼬리 질문 기능을 구성했습니다.

## 2. Problem Definition

- 일반 생성형 AI 답변은 문서 근거가 부족하거나 부정확한 답변을 생성할 수 있습니다.
- AIVLE School 관련 정보는 교육 과정, 지원 자격, 프로젝트, 취업 준비 등 문서 기반 정보 확인이 중요합니다.
- 따라서 문서 검색 기반 질의응답 구조와 사용자 친화적인 학습 인터페이스가 필요했습니다.

## 3. System Flow

```text
사용자 질문 입력
→ 문서 검색
→ 관련 문단 추출
→ LLM 답변 생성
→ 답변 스타일 적용
→ 예상 질문 / 꼬리 질문 제안

## 4. My Role

- AIVLE School 백서의 표, 그래프, 내용 보완
- 학습도우미 인터페이스 구성
- 답변 스타일 설정 옵션 구현
- 문서 기반 예상 질문 기능 구현
- 답변 이후 꼬리 질문 기능 구현
- 사용자 질문 흐름 개선

## 5. Demo

- [Hugging Face Space Demo](https://huggingface.co/spaces/taeyeong7907/streamlit)

## 6. Tech Stack

- Python
- Streamlit
- OpenAI API
- LangChain
- RAG
- PDF / Document Processing

## 7. Key Learning

- RAG는 단순히 LLM을 호출하는 구조가 아니라, 문서 검색과 답변 생성을 연결해 사용자가 신뢰할 수 있는 답변 흐름을 만드는 구조임을 학습했습니다.
- 답변 스타일, 예상 질문, 꼬리 질문 기능은 사용자의 학습 지속성을 높이는 인터페이스 요소로 작동할 수 있습니다.

## 8. Limitations

- 답변 정확도에 대한 정량 평가는 수행하지 않았습니다.
- 따라서 “정확도 향상”이 아니라 “문서 기반 답변 신뢰도 향상을 목표로 설계”했다고 표현합니다.

## 9. Improvement Plan

- 답변 출처 표시 강화
- 문서 검색 결과 근거 문단 표시
- 질문 유형별 프롬프트 분기
- 사용자 피드백 기반 답변 개선
