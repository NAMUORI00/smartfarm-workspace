---
language:
- ko
license: mit
task_categories:
- question-answering
- text-generation
tags:
- agriculture
- wasabi
- smartfarm
- korean
- synthetic
- llm-generated
pretty_name: Wasabi SmartFarm QA Dataset
size_categories:
- n<1K
---

# Wasabi SmartFarm QA Dataset

## Dataset Description

스마트팜 환경에서의 와사비(고추냉이) 재배에 관한 한국어 QA 데이터셋입니다.

### Dataset Summary

- **Size**: 220 QA pairs
- **Language**: Korean (한국어)
- **Domain**: Wasabi cultivation, SmartFarm, Hydroponics
- **Generation Method**: LLM-based synthetic data generation
- **Source Documents**: Web crawled (Wikipedia, Britannica, wasabicrop.co.uk)

### Supported Tasks

- **Question Answering**: 와사비 재배 관련 질문에 대한 답변
- **Information Retrieval**: RAG 시스템 평가
- **Knowledge Extraction**: 농업 도메인 지식 추출

### Languages

- Korean (ko)

## Dataset Structure

### Data Fields

```json
{
  "id": "wasabi_qa_0001",
  "question": "와사비 재배 시 최적 수온은?",
  "answer": "와사비의 최적 수온은 13-17°C입니다...",
  "context": "와사비는 냉수성 작물로...",
  "category": "환경조건",
  "complexity": "basic",
  "source_ids": ["web_wiki_wasabi#c42"],
  "metadata": {
    "model": "gemini-2.5-flash",
    "answer_hint": "13-17°C"
  }
}
```

### Data Splits

| Split | Size |
|-------|------|
| full  | 220  |

### Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| 재배기술 | 48 | 21.8% |
| 양액관리 | 48 | 21.8% |
| 병해충 | 38 | 17.3% |
| 설비장비 | 34 | 15.5% |
| 수확품질 | 30 | 13.6% |
| 환경조건 | 22 | 10.0% |

### Complexity Distribution

| Complexity | Count | Percentage |
|------------|-------|------------|
| basic | 76 | 34.5% |
| intermediate | 88 | 40.0% |
| advanced | 56 | 25.5% |

## Dataset Creation

### Source Data

영문 웹 문서를 크롤링하여 한국어로 번역:
- Wikipedia (Eutrema japonicum)
- Britannica (Wasabi)
- wasabicrop.co.uk (cultivation guides)
- Academic sources

### Generation Methodology

다음 방법론을 적용한 LLM 기반 합성 데이터 생성:

1. **Self-Instruct** (Wang et al., 2023, ACL)
   - Seed questions에서 다양한 질문 자동 생성
   - ROUGE-L 기반 diversity filtering (threshold=0.7)

2. **Evol-Instruct** (Xu et al., 2023, ICLR)
   - In-depth evolving: 복잡성 증가
   - In-breadth evolving: 범위 확장

3. **RAFT** (Zhang et al., 2024, COLM)
   - Context-grounded answer generation
   - Document-based QA pair creation

4. **Prometheus** (Kim et al., 2024, NeurIPS)
   - Rubric-based evaluation
   - Multi-criteria scoring (groundedness, accuracy, completeness)

### Models Used

- **Generator**: `gemini-2.5-flash` (question/answer generation)
- **Judge**: `claude-sonnet-4-5` (quality evaluation)
- **Translation**: `gemini-2.5-flash` (EN→KO)

### Quality Validation

자동 검증 시스템을 통한 품질 보증:

| Metric | Score | Description |
|--------|-------|-------------|
| ROUGE-L Diversity | 0.93 | Question diversity (1.0 = perfect) |
| Groundedness | 0.52 | Answer grounded in context |
| Numerical Consistency | 0.95 | Numbers match source |

## Considerations for Using the Data

### Known Limitations

1. **Synthetic Nature**: All QA pairs are LLM-generated without human authorship
2. **Translation Artifacts**: Korean text translated from English may contain unnatural expressions
3. **Groundedness Issues**: ~48% of answers may contain information not directly in context
4. **No Human Evaluation**: Validation relies on automatic metrics only
5. **Domain-Specific**: Limited to wasabi cultivation; generalization unknown

### Potential Biases

- Source documents primarily from English sources (Western perspective)
- Model biases from gemini-2.5-flash and claude-sonnet-4-5
- Keyword coverage metric may flag legitimate paraphrasing as hallucination

### Recommendations

1. Use as training data for domain-specific QA systems
2. Combine with human-curated data for production systems
3. Verify critical agricultural information with domain experts
4. Consider as supplementary data, not primary source

## Additional Information

### Dataset Curators

Smart Farm RAG Team

### Licensing Information

MIT License

### Citation

```bibtex
@dataset{wasabi_smartfarm_qa_2024,
  title={Wasabi SmartFarm QA Dataset},
  author={Smart Farm RAG Team},
  year={2024},
  publisher={GitHub},
  note={LLM-generated Korean QA dataset for wasabi cultivation}
}
```

### Methodology References

```bibtex
@inproceedings{wang2023self,
  title={Self-Instruct: Aligning Language Models with Self-Generated Instructions},
  author={Wang, Yizhong and others},
  booktitle={ACL},
  year={2023}
}

@inproceedings{xu2023wizardlm,
  title={WizardLM: Empowering Large Language Models to Follow Complex Instructions},
  author={Xu, Can and others},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{zhang2024raft,
  title={RAFT: Adapting Language Model to Domain Specific RAG},
  author={Zhang, Tianjun and others},
  booktitle={COLM},
  year={2024}
}

@inproceedings{kim2024prometheus,
  title={Prometheus: Inducing Fine-grained Evaluation Capability in Language Models},
  author={Kim, Seungone and others},
  booktitle={NeurIPS},
  year={2024}
}
```
