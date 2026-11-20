# 분석용 데이터

MLP·GRU 학습에는 아래 첫 두 파일을 사용한다. 두 파일 모두 2023~2024년의 50개 대여소를 한 시간 간격으로 기록했다. 노트북은 압축된 CSV를 직접 읽고 `station_id`와 `target_time`으로 결합한다.

| 파일 | 행 수 | 용도 |
|---|---:|---|
| [station_hour.csv.gz](station_hour.csv.gz) | 877,200 | 대여소별 시간당 유효 출발 건수, 과거 이용량, 기상·달력 피처 |
| [대여소별_직전_반납.csv.gz](대여소별_직전_반납.csv.gz) | 877,200 | 각 예측 시각 직전 1시간·3시간 동안 해당 대여소에 반납된 건수 |
| [Seoul_public_bicycle.csv](Seoul_public_bicycle.csv) | 81,018 | 2015~2024년 서울 전체 시간별 대여량과 기상·달력 자료. 모델은 이 자료에서 결합한 기상·달력 값을 사용한다. |
| [월별 제외 건수.csv](월별%20제외%20건수.csv) | 24 | 월별 원본 건수와 이용시간·이동거리 필터 적용 후의 건수 대조 |

**대여·반납 기록**은 서울특별시의 공공자전거 대여이력에서 집계했다. 개별 이력이 필요하면 서울 열린데이터광장의 2023년·2024년 ZIP을 내려받을 수 있다. 로컬 실험에 사용한 연도별 ZIP은 이용시간 또는 이동거리가 0인 기록을 제거한 가공본이다. [서울시 공공자전거 대여이력](https://data.seoul.go.kr/dataList/OA-15182/F/1/datasetView.do)

**서울 전체 이용량·기상 자료**는 Kaggle 사용자 lnoahl이 서울 열린데이터광장의 이용량과 기상청 ASOS 관측값을 결합한 자료다. 배포자는 공공누리 제1유형(출처표시)을 명시했다. [Kaggle 데이터셋](https://www.kaggle.com/datasets/lnoahl/seoul-bike-sharing-dataset) · [서울 열린데이터광장](https://data.seoul.go.kr) · [기상청 기상자료개방포털](https://data.kma.go.kr)
