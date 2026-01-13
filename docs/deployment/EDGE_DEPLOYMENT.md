# Edge 환경 배포 가이드

8GB RAM 엣지 디바이스 (NVIDIA Jetson, 저사양 PC) 에서 SmartFarm RAG 시스템을 배포하는 방법을 설명합니다.

## 최소 사양

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| RAM | 8GB | 16GB |
| Storage | 10GB | 20GB |
| CPU | 4코어 | 8코어 |
| OS | Ubuntu 20.04+ / Jetson Linux | Ubuntu 22.04+ |

---

## 빠른 시작 (5분)

### 1. 저장소 클론

```bash
git clone --recurse-submodules https://github.com/NAMUORI00/smartfarm-workspace.git
cd smartfarm-workspace/era-smartfarm-rag
```

### 2. Edge 모드 설치

```bash
python setup.py --mode edge
```

이 명령은 다음을 자동 수행합니다:
- 경량 의존성 설치 (`requirements-edge.txt`)
- MiniLM 임베딩 모델 다운로드 (~90MB)
- Qwen3-0.6B LLM 다운로드 (~400MB)
- 환경변수 설정 (`EMBED_MODEL_ID=minilm`)
- 디렉토리 구조 생성

### 3. 인덱스 빌드 (선택)

사전 빌드된 인덱스가 없으면:

```bash
python scripts/tools/build_index_offline.py \
  --corpus ../dataset-pipeline/output/wasabi_en_ko_parallel.jsonl \
  --output data/index_minilm
```

### 4. 서버 시작

```bash
# 터미널 1: LLM 서버 (llama.cpp)
llama-server -m models/Qwen3-0.6B-Q4_K_M.gguf -c 2048 --port 8080

# 터미널 2: RAG API 서버
python -m core.main

# 터미널 3: (선택) Streamlit UI
streamlit run frontend/streamlit/app.py
```

### 5. 테스트

```bash
# 헬스체크
curl http://localhost:41177/health

# 질의 테스트
curl -X POST http://localhost:41177/query \
  -H "Content-Type: application/json" \
  -d '{"question": "와사비 적정 온도는?", "top_k": 4}'
```

---

## 오프라인 모드

네트워크 없이 또는 LLM 서버 없이 실행:

### 환경변수 설정

```bash
# LLM 없이 검색 결과만 반환
export OFFLINE_MODE=true
python -m core.main
```

### 오프라인 모드 동작

| 상황 | 동작 |
|------|------|
| LLM 서버 연결 실패 | 템플릿 기반 응답 생성 |
| 캐시 히트 | 이전 응답 재활용 |
| 유사 질의 발견 | 유사 질문의 캐시 응답 반환 |

---

## 메모리 최적화

### 경량 임베딩 모델 사용

```bash
# .env 파일
EMBED_MODEL_ID=minilm    # MiniLM-L12 (90MB, 384d)
# EMBED_MODEL_ID=qwen     # Qwen3-Embedding (1.2GB, 1024d) - 비권장
```

### 리랭커 자동 조절

시스템이 가용 메모리에 따라 리랭커를 자동 선택합니다:

| 가용 RAM | 리랭커 |
|----------|--------|
| < 0.8GB | none (리랭킹 비활성화) |
| 0.8~1.5GB | LLM-lite (경량) |
| ≥ 1.5GB | BGE (고품질) |

### FAISS mmap 모드

대용량 인덱스를 메모리에 전부 로드하지 않고 필요한 부분만 로드:

```bash
# .env 파일
DENSE_MMAP=true
```

---

## 디렉토리 구조

```
era-smartfarm-rag/
├── models/                     # GGUF 모델 파일
│   └── Qwen3-0.6B-Q4_K_M.gguf
├── data/
│   ├── index_minilm/          # MiniLM 임베딩 인덱스
│   ├── cache/                  # 응답 캐시
│   └── logs/                   # 쿼리 로그
├── .env                        # 환경 설정
└── core/                       # 애플리케이션 코드
```

---

## 문제 해결

### 메모리 부족

```bash
# 1. 경량 모델 사용 확인
echo $EMBED_MODEL_ID   # minilm 이어야 함

# 2. 불필요한 프로세스 종료
sudo systemctl stop docker   # Docker 사용 안하면

# 3. 스왑 추가 (Jetson)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### LLM 서버 시작 실패

```bash
# 모델 파일 확인
ls -lh models/*.gguf

# 모델 재다운로드
python scripts/download_models.py --edge

# llama.cpp 직접 빌드 (Jetson)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && make -j4
```

### 인덱스 로드 실패

```bash
# 인덱스 파일 확인
ls -la data/index_minilm/

# 인덱스 재빌드
rm -rf data/index_minilm
python scripts/tools/build_index_offline.py --corpus ... --output data/index_minilm
```

---

## 성능 기대값

MiniLM + Qwen3-0.6B 조합 (8GB RAM 환경):

| 메트릭 | 예상 범위 |
|--------|----------|
| 콜드 스타트 | 2-5초 |
| 쿼리 레이턴시 (p95) | 150-300ms |
| 메모리 사용량 | 0.8-1.5GB |
| 처리량 | 3-8 QPS |

---

## Systemd 서비스 등록 (선택)

자동 시작 설정:

```bash
sudo tee /etc/systemd/system/smartfarm-rag.service > /dev/null << 'EOF'
[Unit]
Description=SmartFarm RAG Server
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/era-smartfarm-rag
Environment=EMBED_MODEL_ID=minilm
ExecStart=/usr/bin/python3 -m core.main
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable smartfarm-rag
sudo systemctl start smartfarm-rag
```

---

## 참고

- [ERA RAG README](../../era-smartfarm-rag/README.md)
- [실험 실행 가이드](../paper/EXPERIMENT_GUIDE.md)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
