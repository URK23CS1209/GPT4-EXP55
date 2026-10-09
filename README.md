# 24CS2506 – Building Applications using GPT-4
## Experiment 5: Automated Legal Document Analyzer using GPT-4

---

## 1. Overview
Legal documents are often lengthy, linguistically intricate, and filled with specialized legal jargon, conditional clauses, exceptions, and cross-references. Manually analyzing contracts to extract key responsibilities, payment schedules, duration, liabilities, and termination conditions is both tedious and prone to human oversight.

This project implements an **Automated Legal Document Analyzer using GPT-4**, designed to extract key contractual provisions, analyze risk-bearing clauses, handle document chunking for extensive agreements, and generate abstractive summaries adhering to rigorous prompt constraints without hallucinations.

---

## 2. Key Concepts Covered (Lab Syllabus)

### 1. What is a Legal Document?
A legal document is a formal written instrument that defines legal rights, responsibilities, obligations, agreements, or procedures between two or more parties.
- **Common Examples**:
  - Employment Agreement
  - Non-Disclosure Agreement (NDA)
  - Master Services Agreement (MSA)
  - Commercial Lease / Rental Agreement
  - Sales Agreement
  - Loan Agreement
  - Terms of Service & Privacy Policy

### 2. What is Legal Document Analysis?
Examining a contract to identify:
- **Parties involved** (Entities, titles, representations)
- **Purpose** (Core objective of the agreement)
- **Obligations & Rights** (What each party must or must not do)
- **Financial Terms** (Payment, amounts, milestones, invoicing, currency)
- **Duration & Term** (Start date, duration, renewal mechanisms)
- **Termination Conditions** (Notice periods, breach remedy timelines, convenience exits)
- **Confidentiality Requirements** (Scope of protected data, survival periods)
- **Governing Law & Dispute Resolution** (Arbitration venues, applicable legal code)

### 3. Why is Legal Document Analysis Difficult?
- Complex legal terminology (*indemnification, force majeure, liquidated damages*)
- Long nested sentences and conditional phrasing
- Multi-party cross-references and schedules
- Exceptions, caveats, and survival clauses
- Sheer volume of text in enterprise contracts

### 4. What is Legal Document Summarization?
Distilling lengthy legal agreements into concise summaries that preserve critical legal meaning while eliminating boilerplate verbiage.
$$\text{Long Legal Document} \xrightarrow{\quad\text{GPT-4}\quad} \text{Concise Abstractive Summary}$$

### 5. Components of a Good Legal Analysis Prompt
A reliable prompt enforces zero-hallucination constraints:
- **Role**: `You are a professional legal document analysis assistant.`
- **Task**: `Analyze and summarize the provided legal document.`
- **Information to Extract**: Parties, Purpose, Obligations, Payment terms, Duration, Termination, Important dates.
- **Constraint**: `Use ONLY information present in the document. Do NOT invent facts or extrapolate unsupported details.`
- **Output Format**: Headings and bullet points matching the standard schema.

### 6. Key Clause Identification
The analyzer inspects and categorizes six critical contract clauses:
1. **Confidentiality**: Protection standards, exclusions, survival terms.
2. **Payment**: Schedules, milestones, taxes, interest on default.
3. **Termination**: Notice requirements, for-cause vs for-convenience exit conditions.
4. **Liability**: Aggregate monetary liability caps, exclusions for consequential damages.
5. **Intellectual Property**: Ownership assignment ("work made for hire"), pre-existing IP reservations.
6. **Dispute Resolution**: Mediation, arbitration rules, jurisdiction, and governing law.

### 7. Extractive vs. Abstractive Summarization
- **Extractive Summarization**: Extracts literal, verbatim sentences directly from the source contract.
- **Abstractive Summarization**: GPT-4 understands the legal semantics and generates new, concise prose capturing essential rights, covenants, and dates.

### 8. Chunking Long Documents (Section 9)
Enterprise contracts often exceed standard single-prompt context limits or benefit from section-by-section breakdown:
```
           Legal Document
                 │
                 ▼
     ┌───────────────────────┐
     │ Section Identification│
     └───────────────────────┘
                 │
   ┌─────────────┼─────────────┐
   ▼             ▼             ▼
Chunk 1       Chunk 2       Chunk 3
   │             │             │
   ▼             ▼             ▼
GPT-4 Anal.   GPT-4 Anal.   GPT-4 Anal.
   └─────────────┬─────────────┘
                 ▼
       Combination / Synthesis
                 │
                 ▼
     Consolidated Final Analysis
```

---

## 3. Recommended Output Format

The system produces analysis adhering strictly to the lab manual format:

```text
LEGAL DOCUMENT ANALYSIS
--------------------------------
Document Type:
[Employment Agreement / NDA / Service Agreement / Rental Agreement]

Parties:
[Party 1 and Party 2]

Purpose:
[Objective of agreement]

Duration:
[Term length and start/end dates]

Key Obligations:
• [Obligation 1]
• [Obligation 2]

Payment:
[Monetary terms or consideration]

Termination:
[Notice periods and exit conditions]

Important Dates:
Start Date: [Date]
[Other dates]

Potentially Important Clauses:
• Confidentiality: [Details]
• Payment: [Details]
• Termination: [Details]
• Liability: [Details]
• Intellectual Property: [Details]
• Dispute Resolution: [Details]

Summary:
[Concise abstractive legal summary]
```

---

## 4. System Architecture

```
GPT4-EXP5/
│
├── sample_documents/                     # 5 Pre-packaged legal contracts
│   ├── employment_agreement.txt          # Apex Innovations & Rahul Sharma
│   ├── non_disclosure_agreement.txt      # TechCore & DataVibe mutual NDA
│   ├── service_agreement.txt             # Alpha Retail & CloudScale Technologies
│   ├── rental_agreement.txt              # Commercial Lease (Vikram Oberoi & NextGen)
│   └── long_commercial_contract.txt      # 10-section Master Commercial Contract
│
├── legal_analyzer.py                     # Core extraction, chunking, prompt & GPT-4 engine
├── cli_analyzer.py                       # CLI application with interactive & argument modes
├── app.py                                # Streamlit Web UI with interactive dashboard
├── test_analyzer.py                      # Automated test suite (16 tests, 100% pass)
├── requirements.txt                      # Project dependencies
├── .env.example                          # Environment variable configuration template
└── README.md                             # Lab documentation
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- pip

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment (Optional for Live GPT-4)
Copy `.env.example` to `.env` and set your OpenAI API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
OPENAI_API_KEY=sk-...your-key...
DEFAULT_MODEL=gpt-4o
```
*(Note: If no API key is configured, the application automatically uses its realistic simulation engine so all features, tests, and interfaces work seamlessly offline!)*

---

## 6. How to Run the Applications

### 1. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
- Open `http://localhost:8501` in your browser.
- **Features**:
  - One-click loading of 5 pre-packaged sample contracts.
  - Upload custom `.txt` or `.pdf` agreements.
  - Direct text paste box with real-time word and token estimators.
  - Toggle document chunking with adjustable chunk size and overlap sliders.
  - Interactive tabs: Standard Output Format, Clauses & Obligations cards, Extractive vs Abstractive viewer, Chunking Inspector, and Export (TXT, JSON, Markdown).

### 2. Run via Command-Line Interface (CLI)
#### Interactive Menu:
```bash
python cli_analyzer.py
```

#### Analyze a file directly:
```bash
python cli_analyzer.py --file sample_documents/employment_agreement.txt
```

#### Analyze a long agreement with chunking enabled and save output:
```bash
python cli_analyzer.py --file sample_documents/long_commercial_contract.txt --chunk --output my_analysis.txt
```

---

## 7. Running Automated Tests

Run the complete test suite using pytest:
```bash
pytest test_analyzer.py -v
```

All 16 test cases cover:
- Text extraction & normalization (.txt, .pdf)
- Document chunking algorithms & overlap retention
- Prompt generation & anti-hallucination constraints
- Fallback heuristic analysis across all sample contracts
- Output schema verification
- Safe fallback behavior in offline mode

---

## 8. Summary of Results
The automated legal document analyzer successfully achieves:
1. Fast text extraction from files and strings.
2. Structured extraction of all required legal entities and terms.
3. Modular chunking and synthesis for lengthy commercial contracts.
4. Abstractive summarization preserving core legal covenants without hallucinations.
5. Flexible multi-modal usage via Streamlit Web UI, CLI, and Python library.
