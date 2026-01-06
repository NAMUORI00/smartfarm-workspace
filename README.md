# SmartFarm Workspace

스마트팜 RAG 시스템 통합 개발 환경을 위한 워크스페이스 저장소

## 클론

```bash
git clone --recurse-submodules https://github.com/NAMUORI00/smartfarm-workspace.git
```

기존 클론 후 submodule 가져오기:
```bash
git submodule update --init --recursive
```

## 프로젝트 구성

| 폴더 | 설명 | 저장소 |
|------|------|--------|
| `era-smartfarm-rag/` | RAG 서버 (엣지 환경용 Qwen3 + llama.cpp) | [era-smartfarm-rag-qwen3-edge](https://github.com/NAMUORI00/era-smartfarm-rag-qwen3-edge) |
| `dataset-pipeline/` | LLM-as-a-Judge 데이터셋 생성 파이프라인 | [smartfarm-dataset-pipeline](https://github.com/NAMUORI00/smartfarm-dataset-pipeline) |

## 워크플로우

### 각 프로젝트에서 작업 후 커밋

```bash
# RAG 서버 작업
cd era-smartfarm-rag
# ... 수정 ...
git add . && git commit -m "Update" && git push

# Dataset 파이프라인 작업
cd ../dataset-pipeline
# ... 수정 ...
git add . && git commit -m "Update" && git push

# 워크스페이스 동기화 (submodule 참조 업데이트)
cd ..
git add .
git commit -m "Update submodules"
git push
```

### Submodule 최신 버전으로 업데이트

```bash
git submodule update --remote --merge
```

## 개별 프로젝트만 사용

워크스페이스 없이 각 프로젝트를 독립적으로 사용할 수 있습니다:

```bash
# RAG 서버만 필요할 때
git clone https://github.com/NAMUORI00/era-smartfarm-rag-qwen3-edge.git

# Dataset 파이프라인만 필요할 때
git clone https://github.com/NAMUORI00/smartfarm-dataset-pipeline.git
```
