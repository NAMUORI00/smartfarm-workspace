# DAT (Dynamic Adaptive 3-Channel) Formal Definition

이 문서는 SmartFarm 벤치마크에 사용한 DAT의 수식 정의와 운용 제약을 정식화한다.

## 1. 채널별 랭킹
질문 \(q\)에 대해 세 채널의 순위 목록:
\[
R_d(q), R_s(q), R_g(q)
\]
각각 dense, sparse, graph 채널에서 상위 \(k'\)개 문서 집합을 반환.

## 2. 동적 가중치 함수
DAT의 질의별 가중치 벡터:
\[
\mathbf{w}(q)=\left(w_d(q), w_s(q), w_g(q)\right)
\]
with
\[
\sum_{c\in\{d,s,g\}} w_c(q)=1,\quad w_c(q)\ge 0.
\]

전역 최소/최대 제약:
\[
\alpha \le w_c(q)\le \beta,\quad
\alpha=0.10,\ \beta=0.80.
\]

시간적 갱신 제약(연속 가중치 변화 제한):
\[
|w_c^{(t)}(q)-w_c^{(t-1)}(q)|\le \Delta,\quad \Delta=0.15.
\]

초기값은 균등 또는 학습된 기본값 \(\mathbf{w}_0=(0.34,0.33,0.33)\) 사용.

## 3. 가중치 기반 융합
문서 \(d\)에 대한 최종 점수:
\[
S(q,d)=\sum_{c\in\{d,s,g\}} w_c(q)\sum_{r=1}^{k'}\frac{\mathbb{1}[d\in R_c^r(q)]}{K+r},
\]
\(K=60\), \(R_c^r\)은 채널 \(c\)의 \(r\)순위 문서 집합.

최종 랭킹:
\[
R_{\text{DAT}}(q)=\operatorname{rank}\left(S(q,d)\right)_{d\in \cup_c R_c(q)}.
\]

## 4. 구간 분리/적응 선택
메타 규칙 \(\phi(q)\)로 질의 속성을 추출한다.
\[
\phi(q)\in \mathcal{Q}=\{\text{short},\text{medium},\text{long}\}\times\{\text{relational},\text{non-relational}\}\times\{\text{numeric},\text{non-numeric}\}
\]
각 세그먼트 \(s\in\mathcal{S}\)마다 후보 가중치 \(\mathbf{w}_s\)를 학습하여
\[
\mathbf{w}(q)=\mathbf{w}_{\phi(q)}.
\]

세그먼트 커버리지는
\[
\text{coverage}(s)=\frac{|Q_s|}{|Q_{\text{tune}}|},
\]
신뢰도는 \( \rho_s=\min(1,\ 0.5+0.5\cdot \text{coverage}(s))\)로 계산.

## 5. 학습/튜닝 목표
튜닝 샘플 집합 \(D_{\text{train}}\)에서 가중치 후보 \(w\)와 depth \(h\)를 정하면:
\[
\mathcal{J}(w,h)=\overline{\mathrm{nDCG@}k}(w,h) - \lambda\cdot \sigma_{\text{CV}}(w,h),
\]
\[
\lambda=0.25,\quad (w,h)=\arg\max \mathcal{J}(w,h),
\]
\(\sigma_{\text{CV}}\)는 CV fold별 성능 표준편차.

## 6. 학습 분할 및 누수 방지
공식 split:
\[
Q_{\text{all}} = Q_{\text{train}}\cup Q_{\text{val}}\cup Q_{\text{test}},
\]
\[
Q_{\text{train}}\cap Q_{\text{val}}=Q_{\text{train}}\cap Q_{\text{test}}=Q_{\text{val}}\cap Q_{\text{test}}=\varnothing.
\]

DAT 튜닝에는 \(Q_{\text{train}}\)와 \(Q_{\text{val}}\)만 사용:
\[
\text{fit DAT} = \arg\max \mathcal{J}(w,h;Q_{\text{train}}\cup Q_{\text{val}}),
\]
최종 지표 보고는 오직 \(Q_{\text{test}}\)에서 수행.

`run_main_eval`/`paper_eval`에서 split manifold(공식/재분할)을 저장:
- `output/splits/<dataset>_split_manifest.json`

## 7. Guardrail/quality gate
품질 게이트:
- \(Q_{\text{tune}}\) 크기가 임계치 미만이면 DAT fallback:
  \[
  \mathbf{w}(q)=\mathbf{w}_0.
  \]
기본값: `DAT_MIN_PROFILE_QUALITY_QUERIES=300`.

세그먼트 품질 미만시 기본 융합으로 폴백.

## 8. 실패 대응
실패 시 동작:
- 채점 실패 항목은 `metric_applicability`로 추적
- `no_rag`는 IR/RAGAS N/A 정책으로 분리
- judge 호출 실패가 누적되면 fail-fast를 적용해 조기 종료 후 seed 레벨 결과 반환
