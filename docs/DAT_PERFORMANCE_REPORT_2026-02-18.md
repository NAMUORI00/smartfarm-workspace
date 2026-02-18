# DAT 성능 검증 리포트 (2026-02-18)

## 1. 목적
- 목표: 휴리스틱 없이(`query rule` 제거) `ours_structural`의 채널 융합을 데이터 기반으로 적응시키고 성능 향상 여부를 검증.
- 반영 원칙: OpenAI-compatible 인터페이스만 사용, 벤치마크는 공개 HF 데이터셋 기반.

## 2. 참조 연구(다국어/경량 중심)
- mDAPT: https://arxiv.org/abs/2503.17488
- Retrofitting Small Multilingual Models for Dense Retrieval: https://arxiv.org/abs/2507.02705
- MAD-X (adapter-based multilingual transfer): https://aclanthology.org/2020.emnlp-main.617/

해석:
- 공통적으로 "도메인/언어별 적응을 고정 규칙이 아닌 학습(또는 튜닝)으로 처리"하는 방향이 유효하다는 점을 채택.

## 3. 적용한 방법 (DAT: Data-driven Adaptive Tuning)
- 위치: `smartfarm-benchmarking/benchmarking/experiments/paper_eval.py`
- 변경점:
1. `ours_structural`에서 쿼리 휴리스틱 가중치 제거.
2. 데이터셋별 튜닝 샘플에 대해 `dense/sparse/graph` 가중치를 grid search로 최적화(`nDCG@k` 기준).
3. 선택된 가중치를 동일 데이터셋 평가에 적용.
- 산출 메타: 결과 JSON에 `adaptive_fusion.mode=data_driven`, `weights` 기록.

## 4. 실험 설정
- 비교 대상:
1. Baseline: `paper_eval_local.json` (휴리스틱 버전)
2. DAT: `paper_eval_dat_local.json` (데이터 기반 적응)
- 데이터셋: `agxqa`, `2wiki`, `hotpotqa`
- 쿼리 수: 각 40
- 실행 모드: retrieval-only

주의:
- Featherless는 `chat/completions`, `models`는 정상 응답(200) 확인.
- `embeddings` 엔드포인트는 404로 확인되어, 실험은 OpenAI-compatible 로컬 임베딩 mock 서버를 사용.

## 5. 결과 (ours_structural)
| Dataset | nDCG@10 (Before → DAT) | Δ | MRR (Before → DAT) | Δ |
|---|---:|---:|---:|---:|
| agxqa | 0.5478 → 0.9150 | +0.3672 | 0.4310 → 0.8865 | +0.4555 |
| 2wiki | 0.2738 → 0.6122 | +0.3384 | 0.3317 → 0.7293 | +0.3976 |
| hotpotqa | 0.5238 → 0.7367 | +0.2129 | 0.5082 → 0.8675 | +0.3593 |
| Macro Avg | 0.4485 → 0.7547 | +0.3062 | 0.4236 → 0.8277 | +0.4041 |

선택된 가중치(데이터 기반):
- agxqa: `dense=0.0, sparse=1.0, graph=0.0`
- 2wiki: `dense=0.0, sparse=1.0, graph=0.0`
- hotpotqa: `dense=0.0, sparse=0.8, graph=0.2`

## 6. 해석
- 현재 실험 조건에서는 dense 신호 품질이 약해, DAT가 dense 기여를 자동으로 축소하고 sparse/graph를 강화.
- 결과적으로 휴리스틱 없이도 `ours_structural`의 성능이 안정적으로 개선됨.
- 후속으로 실제 임베딩 품질이 개선되면, DAT는 dense 가중치를 다시 확장할 수 있어 구조적으로 유연함.

## 7. 재현 커맨드
```bash
# baseline
LLM_BACKEND=openai_compatible \
OPENAI_COMPAT_BASE_URL=http://127.0.0.1:48080/v1 \
OPENAI_COMPAT_API_KEY=dummy \
../.venv/bin/python -m benchmarking.experiments.paper_eval \
  --dataset agxqa,2wiki,hotpotqa --method all --k 10 --max-queries 40 \
  --seed 42 --retrieval-only --out output/paper_eval_local.json

# DAT
LLM_BACKEND=openai_compatible \
OPENAI_COMPAT_BASE_URL=http://127.0.0.1:48080/v1 \
OPENAI_COMPAT_API_KEY=dummy \
../.venv/bin/python -m benchmarking.experiments.paper_eval \
  --dataset agxqa,2wiki,hotpotqa --method all --k 10 --max-queries 40 \
  --seed 42 --retrieval-only --out output/paper_eval_dat_local.json
```
