# DAT 성능 검증 리포트 (2026-02-18)

## 1. 목적
- 휴리스틱 없는 DAT(`ours_structural`)를 유지한 상태에서 성능 게이트 통과 가능성을 검증.
- 운영 원칙 고정:
1. LLM 경로: Featherless (OpenAI-compatible)
2. 임베딩 경로: Hugging Face Inference API `feature-extraction`
3. 임베딩 기본모델: `BAAI/bge-m3`(1024d, MIT, multilingual)로 상향

## 2. 참조 연구(다국어/경량 중심)
- mDAPT: https://arxiv.org/abs/2503.17488
- Retrofitting Small Multilingual Models for Dense Retrieval: https://arxiv.org/abs/2507.02705
- MAD-X: https://aclanthology.org/2020.emnlp-main.617/

## 3. 코드 반영 사항
- 파일: `smartfarm-benchmarking/benchmarking/experiments/paper_eval.py`
1. DAT 가중치 탐색 해상도 상향(`grid_step 0.1 -> 0.05`)
2. 튜닝/평가 분할에서 순서편향 제거(데이터셋별 deterministic shuffle)
3. 퍼채널 후보 랭크 심도는 기존 deep candidate 유지(`k*4`)

- 파일: `smartfarm-search/core/embeddings/huggingface_api_provider.py`
- 파일: `smartfarm-ingest/pipeline/embeddings/huggingface_api_provider.py`
- 파일: `smartfarm-benchmarking/benchmarking/embeddings/huggingface_api_provider.py`
1. 임베딩 provider를 HF API로 단일화
2. 요청 timeout + retry + 캐시 적용(휴리스틱 fallback 없음)
3. 기본 임베딩 모델을 `sentence-transformers/distiluse-base-multilingual-cased-v2`(512d)에서 `BAAI/bge-m3`(1024d)로 교체

## 4. 실험 설정 (1차 빠른 검증 러닝)
- 데이터셋: `agxqa`, `2wiki`, `hotpotqa`
- Seed: `42, 52, 62`
- `max_queries=20`, `k=10`, retrieval-only
- 산출물:
1. `smartfarm-benchmarking/output/paper_eval_main.json`
2. `smartfarm-benchmarking/output/comparison_report.json`
3. `smartfarm-benchmarking/output/paper_tables_ieee.md`
4. `smartfarm-benchmarking/output/paper_tables_ieee.tex`

## 5. 결과 (Strongest baseline 대비 Ours 절대차)
> 아래 수치는 모델 교체 이전 러닝 기준이며, `BAAI/bge-m3` 기준 재측정이 필요함.
| Dataset | Δ nDCG@10 (보강 전) | Δ nDCG@10 (보강 후) | Δ MRR (보강 전) | Δ MRR (보강 후) |
|---|---:|---:|---:|---:|
| agxqa | -0.1855 | -0.0279 | -0.2109 | -0.0365 |
| 2wiki | -0.0393 | +0.0050 | -0.0507 | +0.0104 |
| hotpotqa | -0.0189 | -0.0312 | -0.0248 | -0.0766 |
| Macro Avg | -0.0812 | -0.0180 | -0.0955 | -0.0342 |

- 현재 `comparison_report.json` 기준 `overall_pass=false`
- 즉, 게이트 조건(`+0.02 절대개선 + 유의성`)은 아직 미충족

## 6. 해석
- DAT 보강으로 agxqa/2wiki는 유의미하게 개선되었지만, hotpotqa에서 하락.
- 현 단계는 “전반적 격차 축소”까지 달성했고 “게이트 통과”는 미달.
- 다음 사이클은 데이터셋별 분리 DAT 프로파일(학습/적용 분기)과 채널 신뢰도 보정을 우선 검토.

## 7. 재현 커맨드
```bash
LLM_BACKEND=openai_compatible \
OPENAI_COMPAT_BASE_URL=https://api.featherless.ai/v1 \
OPENAI_COMPAT_API_KEY=*** \
OPENAI_COMPAT_MODEL=openai/gpt-oss-120b \
HF_TOKEN=*** \
../.venv/bin/python -m benchmarking.experiments.run_main_eval \
  --dataset agxqa,2wiki,hotpotqa \
  --method bm25_only,dense_only,rrf,graph_only,lightrag,ours_structural \
  --seeds 42,52,62 \
  --max-queries 20 \
  --bootstrap-resamples 1000 \
  --out-json output/paper_eval_main.json \
  --out-csv output/paper_eval_main.csv \
  --comparison-out output/comparison_report.json
```
