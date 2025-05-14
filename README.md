# Concordance-Agent: LLM-Based Radiology–Pathology Concordance Evaluation

![image](https://github.com/user-attachments/assets/2e5ead9e-5704-4a1c-9c17-df24f3749157)


## Project Overview  
Automatically checks **concordance** vs. **discordance** between ultrasound imaging assessments (TI-RADS scores) and biopsy pathology outcomes (benign vs. malignant). Implemented as a **multi-agent LLM system** running on an on-premise PACS environment, this project demonstrates how specialized AI agents can collaborate (using open protocals) to flag discrepancies and generate actionable clinician alerts.

## Key Features  
- **Multi-Agent Design**  
  - **Case Coordinator Agent** supervises workflows, assigning tasks via A2A requests
  - **Radiologist Agent** extracts TI-RADS/BI-RADS findings  
  - **Pathologist Agent** interprets biopsy results  
  - **Concordance Evaluator Agent** applies clinical rules and LLM reasoning  
  - **Notifier Agent** drafts clinician-ready alerts (or use tools instead of agents)
- **Open Protocols**  
  - **Model Context Protocol (MCP)** for secure, unified data access  
  - **Agent2Agent Protocol (A2A)** for standardized, authenticated inter-agent communication  
- **Synthetic Data Pipeline**  
  - Simple CSV format: `case_id, tirads_score, biopsy_result`  
  - Easily extensible to JSON Lines for richer metadata  
- **Extensible & Future-Ready**  
  - Easily integrates free-text report parsing  
  - Scalable to additional modalities (e.g., mammography, genomics)

---

## Design Rationale  
1. **Modularity & Reliability**  
   - Specialized agents mirror real clinical roles (radiologist, pathologist, QA reviewer), reducing error risk.  
2. **Transparency & Explainability**  
   - Concordance Evaluator produces natural-language rationales, supporting auditability and trust.  
3. **Efficient Context Handling**  
   - MCP serves as external memory, avoiding prompt bloat and context-window limits.  
4. **Secure Collaboration**  
   - A2A enforces authentication, task tracking, and capability discovery, ensuring robust on-prem deployment.

---

## Getting Started

### Prerequisites  
- Python 3.9+  
- `graphviz` (for architecture visualization)  
- Network access to your on-prem MCP server and PACS/LIS connectors

### Installation  
```bash
git clone https://github.com/Michael-Susu12138/Concordance-Agent.git
cd rad-path-concordance
pip install -r requirements.txt
