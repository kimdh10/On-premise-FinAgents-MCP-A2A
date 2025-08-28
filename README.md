# 온프레미스 금융투자정보 분석을 위한 Hybrid MCP-A2A 멀티에이전트 시스템

본 저장소는 **온프레미스 환경**에서 금융 투자 정보를 자동으로 분석하고  
보고서를 생성하기 위한 **Hybrid MCP(Model Context Protocol) + A2A(Agent-to-Agent)** 기반  
**멀티에이전트 시스템**의 구현체입니다.  

MCP를 통해 외부 데이터 수집 및 도구 호출을 표준화하고,  
A2A 구조를 통해 에이전트 간 협업을 가능하게 하여  
실시간 금융 분석 보고서를 자동으로 생성할 수 있습니다.

---

##  주요 특징
- **온프레미스 실행**: 외부 클라우드 의존 없이 내부 환경에서 안전하게 동작  
- **MCP 통합**: 주가/재무/뉴스 데이터를 안전하게 수집 (예: Yahoo Finance MCP)
- **A2A 협업**: 생성(Generation), 검색(Retrieval), 검증(Verification) 에이전트의 역할 분담  
- **자동 보고서 생성**: 사용자 질의 → 데이터 수집 → 검증 → 최종 HWP 보고서 출력  
- **확장성**: 새로운 데이터 소스, 맞춤형 LLM, 추가 에이전트로 손쉬운 확장 가능  

---

## MCP
![MCP](img/MCP.png)

---

## A2A
![A2A](img/A2A.png)


---

##  시스템 아키텍처
본 시스템은 **MCP-A2A 하이브리드 파이프라인**으로 구성됩니다:

![Overall Pipeline](img/OverallPipeline.png)

1. **생성 에이전트 (Generation Agent)**  
   - 사용자 요청을 해석하고 필요한 데이터 및 분석 단계를 설계  
2. **검색 에이전트 (Retrieval Agent)**  
   - MCP 도구 호출을 통해 주가/뉴스/재무 데이터를 수집  
   - 필요 시 RAG(Retrieval-Augmented Generation) 수행  
3. **검증 에이전트 (Verification Agent)**  
   - 보고서 초안을 검토하고 인용/일관성 오류를 보정  
   - 필요 시 반복적으로 수정 요청  
4. **보고서 생성 (Report Generation)**  
   - 최종 결과를 HWP 문서(MCP-HWP)를 통해 자동 생성  

---

##  구현 환경
| 항목        | 사양 |
|-------------|------|
| CPU         | Intel Core i5-12600K |
| RAM         | 32 GB |
| GPU         | NVIDIA RTX A5000 (24GB VRAM) |
| OS          | Windows 11 |
| Python      | 3.11 |
| Framework   | LangGraph 0.6.4, LangChain 0.3.26, ollama 0.5.3 |
| LLM 모델    | Generation: Qwen2.5-14B-Instruct<br>Search: Qwen2.5-7B-Instruct-q4_K_M<br>Verification: Qwen2.5-14B-Instruct |
| MCP 도구    | Yahoo Finance MCP, HWP-MCP |

---

##  코드 구조
```
.
├── main.py               # 메인 파이프라인
├── rag_module.py         
├── m2_reducer.py
├── requirements.txt      # 실행 환경 의존성
└── README.md             # 프로젝트 설명 문서
```

---

##  결과
- **데모**: Microsoft(MSFT)에 대한 자동 보고서 생성 실험  
- **출력**: 주가, 뉴스, 재무 데이터가 포함된 HWP 형식 보고서  
- **실험 결론**:  
  - MCP-A2A 기반 온프레미스 멀티에이전트 협업이 실현 가능함을 확인  
  - 보고서 품질 향상을 위해 맞춤형 LLM 적용 필요성 확인  
![QueryInterface](img/QueryInterface.png)
![ToolInvocation](img/ToolInvocation.png)
![Report(HWP)View](img/Report(HWP)View.png)
---

##  향후 연구 방향
- 도메인 특화 LLM 파인튜닝 적용  
- 다중 시장/다국어 데이터 확장  
- 에이전트 워크플로우 모니터링을 위한 실시간 대시보드 구축  

---

##  연구진
- 박시형 (Dong-Seoul University)  
- 김두홍 (Dong-Seoul University)    

---

##  참고 문헌 
- Xinyi Hou et al., *Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions*, arXiv:2503.23278  
- Awid Vaziry et al., *Towards Multi-Agent Economies: Enhancing the A2A Protocol*, arXiv:2507.19550  

---
