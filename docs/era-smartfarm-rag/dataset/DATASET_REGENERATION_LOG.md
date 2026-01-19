# Dataset Regeneration Log

## 목적
- 질문-문맥 lexical leakage를 줄여 hybrid retrieval 평가 편향을 완화한다.
- 동일/중복 문서 ID를 정답 라벨에 함께 포함해 dense/hybrid의 공정성을 높인다.
- 개선 전/후 QA 데이터셋을 재생성해 성능 변화를 추적한다.

## API 설정
- OPENAI_BASE_URL: http://ddns.namuori.net:8317/v1
- API_KEY: 환경변수 사용 (값은 저장하지 않음)
- Generator Model: gemini-3-flash-preview

## 개선 전 (Baseline) 재생성
- 명령:
  - `python -m dataset_pipeline.cli generate-qa -i output/wasabi_en_ko_parallel.jsonl -o output/wasabi_qa_dataset_v2_baseline.jsonl`
- 옵션:
  - lexical filter: off
  - paraphrase: off
  - multi-source: off
- 결과 파일:
  - `dataset-pipeline/output/wasabi_qa_dataset_v2_baseline.jsonl`

## 개선 후 (Improved) 재생성
- 명령:
  - `python -m dataset_pipeline.cli generate-qa -i output/wasabi_en_ko_parallel.jsonl -o output/wasabi_qa_dataset_v2_improved.jsonl --lexical-threshold 0.2 --paraphrase --multi-source`
- 옵션:
  - lexical filter: on (char-3gram Jaccard >= 0.2 제거)
  - paraphrase: on (질문 재표현)
  - multi-source: on (동일 텍스트 문서 ID를 source_ids에 포함)
- 결과 파일:
  - `dataset-pipeline/output/wasabi_qa_dataset_v2_improved.jsonl`

## 변경 사항 요약 (코드)
- `dataset-pipeline/src/dataset_pipeline/cli.py`
  - generate-qa에 lexical filter/paraphrase/multi-source 옵션 추가
  - 질문-문맥 overlap 필터 및 메타데이터 기록 추가

## 다음 단계
- 개선 전/후 QA로 RAG 벤치마크 재실행
- sparse-only 우세 완화 여부 확인(MRR/NDCG/Recall)

## 실행 결과
- 실행 일시: 2026-01-19 11:57:23
- Baseline QA 수: 220
- Improved QA 수: 220
- 비고: 개선 후 생성은 필터/패러프레이즈로 인해 시간이 길어져 resume 재실행 필요
