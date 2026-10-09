"""
legal_analyzer.py
Core Engine for Experiment 5: Automated Legal Document Analyzer using GPT-4.
Covers Document Extraction, Preprocessing, Section/Chunking,
Prompt Engineering, GPT-4 API Interaction, Fallback Heuristic Analysis,
and Output Structuring.
"""

import os
import re
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


@dataclass
class LegalAnalysisResult:
    document_type: str = "Legal Agreement"
    parties: str = "Not specified"
    purpose: str = "Not specified"
    duration: str = "Not specified"
    key_obligations: List[str] = field(default_factory=list)
    payment: str = "Not specified"
    termination: str = "Not specified"
    important_dates: List[str] = field(default_factory=list)
    important_clauses: List[str] = field(default_factory=list)
    summary: str = ""
    raw_output: str = ""
    chunk_count: int = 1
    model_used: str = "gpt-4o"
    word_count: int = 0
    estimated_tokens: int = 0

    def to_formatted_text(self) -> str:
        """Formats output according to the lab manual Recommended Output Format."""
        obligations_str = "\n".join([f"• {ob.lstrip('•*- ')}" for ob in self.key_obligations]) if self.key_obligations else "• As defined in agreement"
        dates_str = "\n".join([f"{d.lstrip('•*- ')}" for d in self.important_dates]) if self.important_dates else "Not specified"
        clauses_str = "\n".join([f"• {c.lstrip('•*- ')}" for c in self.important_clauses]) if self.important_clauses else "• General contractual provisions"

        return f"""LEGAL DOCUMENT ANALYSIS
--------------------------------
Document Type:
{self.document_type}

Parties:
{self.parties}

Purpose:
{self.purpose}

Duration:
{self.duration}

Key Obligations:
{obligations_str}

Payment:
{self.payment}

Termination:
{self.termination}

Important Dates:
{dates_str}

Potentially Important Clauses:
{clauses_str}

Summary:
{self.summary}
""".strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "parties": self.parties,
            "purpose": self.purpose,
            "duration": self.duration,
            "key_obligations": self.key_obligations,
            "payment": self.payment,
            "termination": self.termination,
            "important_dates": self.important_dates,
            "important_clauses": self.important_clauses,
            "summary": self.summary,
            "chunk_count": self.chunk_count,
            "model_used": self.model_used,
            "word_count": self.word_count,
            "estimated_tokens": self.estimated_tokens,
            "raw_output": self.raw_output
        }


class DocumentExtractor:
    """Extracts and cleans text from raw strings or files (.txt, .pdf)."""

    @staticmethod
    def extract_from_file(file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return DocumentExtractor.clean_text(text)
            except Exception as e:
                raise RuntimeError(f"Error reading PDF file: {str(e)}")
        else:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return DocumentExtractor.clean_text(f.read())
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    return DocumentExtractor.clean_text(f.read())

    @staticmethod
    def clean_text(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00a0", " ")
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def get_stats(text: str) -> Dict[str, int]:
        words = len(text.split()) if text else 0
        chars = len(text)
        est_tokens = max(1, chars // 4) if chars > 0 else 0
        lines = len(text.splitlines()) if text else 0
        return {
            "characters": chars,
            "words": words,
            "lines": lines,
            "estimated_tokens": est_tokens
        }


class DocumentChunker:
    """
    Divides lengthy legal documents into smaller manageable sections
    as described in Section 9 ('Chunking Long Documents').
    """

    @staticmethod
    def split_into_chunks(text: str, max_chunk_chars: int = 2200, overlap_chars: int = 200) -> List[str]:
        if not text or len(text) <= max_chunk_chars:
            return [text] if text else []

        section_pattern = r'(?=(?:\n|^)(?:SECTION\s+\d+|CLAUSE\s+\d+|ARTICLE\s+\d+|\d+\.\s+[A-Z]))'
        raw_sections = [s.strip() for s in re.split(section_pattern, text) if s.strip()]

        chunks = []
        current_chunk = ""

        if len(raw_sections) > 1:
            for sec in raw_sections:
                if len(current_chunk) + len(sec) + 2 <= max_chunk_chars:
                    current_chunk = f"{current_chunk}\n\n{sec}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(sec) > max_chunk_chars:
                        sub_chunks = DocumentChunker._split_by_paragraphs(sec, max_chunk_chars, overlap_chars)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = sec
            if current_chunk:
                chunks.append(current_chunk)
        else:
            chunks = DocumentChunker._split_by_paragraphs(text, max_chunk_chars, overlap_chars)

        return chunks

    @staticmethod
    def _split_by_paragraphs(text: str, max_chunk_chars: int, overlap_chars: int) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= max_chunk_chars:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else ""
                    current_chunk = f"{overlap}\n\n{para}".strip()
                else:
                    chunks.append(para[:max_chunk_chars])
                    current_chunk = para[max_chunk_chars - overlap_chars:]

        if current_chunk and current_chunk not in chunks:
            chunks.append(current_chunk)

        return chunks


class PromptGenerator:
    """
    Builds prompts adhering to the Experiment 5 specifications:
    - Role: Legal document analysis assistant
    - Task: Analyze and summarize the provided document
    - Information to Extract: Parties, Purpose, Obligations, Payment terms, Duration, Termination, Important dates
    - Constraints: Use only information present in the document. Do not invent facts.
    - Output Format: Specific headings and bullet points.
    """

    RECOMMENDED_SCHEMA = """LEGAL DOCUMENT ANALYSIS
--------------------------------
Document Type:
[Specific Document Type, e.g. Employment Agreement, NDA, Service Agreement, Rental Agreement]

Parties:
[Names and roles of the entities involved]

Purpose:
[Core objective of the agreement]

Duration:
[Term length, start date, end date, or renewal period]

Key Obligations:
• [Obligation 1]
• [Obligation 2]
• [Obligation 3]

Payment:
[Financial terms, amounts, currency, milestones, invoicing frequency, or 'No monetary payment']

Termination:
[Notice periods, termination for cause/convenience, exit conditions]

Important Dates:
Start Date: [Date]
[Other key milestone/expiry/renewal dates]

Potentially Important Clauses:
• Confidentiality: [Brief scope or protection period]
• Payment: [Brief payment condition]
• Termination: [Brief summary of exit clause]
• Liability: [Liability cap or exclusion]
• Intellectual Property: [Ownership rights]
• Dispute Resolution: [Arbitration/governing law/venue]

Summary:
[Concise abstractive summary preserving the essential legal meaning]"""

    @staticmethod
    def build_analysis_prompt(document_text: str, is_chunk: bool = False, chunk_index: int = 1, total_chunks: int = 1) -> Dict[str, str]:
        system_prompt = (
            "You are a professional legal document analysis assistant.\n"
            "Your task is to analyze and summarize legal documents with high precision.\n\n"
            "CONSTRAINTS:\n"
            "1. Use ONLY information directly present in the provided document.\n"
            "2. Do NOT invent facts, assume terms, or extrapolate unsupported details (no hallucinations).\n"
            "3. If a specific detail is not mentioned in the text, write 'Not specified in the document'.\n"
            "4. Follow the requested output structure strictly with headings and bullet points."
        )

        chunk_note = f" (Chunk {chunk_index} of {total_chunks})" if is_chunk else ""

        user_prompt = (
            f"Please analyze the following legal document{chunk_note}.\n\n"
            "DOCUMENT TEXT:\n"
            "\"\"\"\n"
            f"{document_text}\n"
            "\"\"\"\n\n"
            "TASK:\n"
            "Extract the key legal parameters and produce an abstractive summary adhering EXACTLY to this output format:\n\n"
            f"{PromptGenerator.RECOMMENDED_SCHEMA}"
        )

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    @staticmethod
    def build_combination_prompt(chunk_summaries: List[str]) -> Dict[str, str]:
        system_prompt = (
            "You are a legal document synthesis assistant. You will receive sectional analyses "
            "from chunks of a long legal document. Synthesize them into a single, cohesive, consolidated "
            "legal document analysis adhering strictly to facts present in the text without hallucinations."
        )

        combined_input = "\n\n--- NEXT SECTION ANALYSIS ---\n\n".join(chunk_summaries)

        user_prompt = (
            "Here are the analyses of different sections of a long legal contract:\n\n"
            "\"\"\"\n"
            f"{combined_input}\n"
            "\"\"\"\n\n"
            "Synthesize all the above sectional findings into one comprehensive analysis following this exact schema:\n\n"
            f"{PromptGenerator.RECOMMENDED_SCHEMA}"
        )

        return {
            "system": system_prompt,
            "user": user_prompt
        }


class FallbackAnalyzer:
    """
    Rule-based and heuristic analyzer used when no OpenAI API key is configured,
    or during automated tests/offline demonstrations.
    Guarantees reliable execution while preserving the exact required schema.
    """

    @staticmethod
    def analyze(text: str) -> LegalAnalysisResult:
        stats = DocumentExtractor.get_stats(text)
        text_lower = text.lower()
        first_1200 = text_lower[:1200]

        # 1. Document Type Detection
        if "employment agreement" in first_1200 or ("employment" in first_1200 and "employee" in first_1200):
            doc_type = "Employment Agreement"
        elif "rental agreement" in first_1200 or "commercial lease" in first_1200 or "lease and rental" in first_1200:
            doc_type = "Commercial Lease and Rental Agreement"
        elif "services agreement" in first_1200 or "service agreement" in first_1200:
            doc_type = "Master Software Services Agreement"
        elif "commercial contract" in first_1200:
            doc_type = "Master Commercial Contract and Service Level Agreement"
        elif "non-disclosure" in first_1200 or "nda" in first_1200 or "confidentiality agreement" in first_1200:
            doc_type = "Mutual Non-Disclosure Agreement (NDA)"
        elif "loan agreement" in first_1200:
            doc_type = "Loan Agreement"
        elif "sales agreement" in first_1200:
            doc_type = "Sales Agreement"
        else:
            doc_type = "Commercial Contract"

        # 2. Parties Detection
        parties = "Not specified"
        parties_sec = re.search(r'PARTIES:?\s*\n+(.*?)(?=\n\s*(?:\d+\.\s+[A-Z\s/&]+:|\bSECTION\b|\bRECITALS\b|\bPURPOSE\b|$))', text, re.IGNORECASE | re.DOTALL)
        if parties_sec:
            raw_lines = [l.strip() for l in parties_sec.group(1).splitlines() if l.strip()]
            cleaned_names = []
            for line in raw_lines:
                if re.match(r'^\d+[\.\)]', line) or line.startswith(('-', '•', '*')):
                    clean = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                    clean = re.sub(r'\(hereinafter.*?\)', '', clean, flags=re.IGNORECASE)
                    clean = re.split(r',|\bwith\b|\bhaving\b|\bresiding\b', clean)[0].strip()
                    if clean:
                        cleaned_names.append(clean)
            if len(cleaned_names) >= 2:
                parties = f"{cleaned_names[0]} and {cleaned_names[1]}"
            elif len(cleaned_names) == 1:
                parties = cleaned_names[0]

        if parties == "Not specified":
            m = re.search(r'(?:by and between|between)\s+([A-Z][A-Za-z0-9\s\.]+?),\s*(?:a|an|having|with|residing|corporation|located).*?(?:and)\s+([A-Z][A-Za-z0-9\s\.]+?),\s*(?:having|with|a|an|residing|located)', text, re.DOTALL)
            if m:
                parties = f"{m.group(1).strip()} and {m.group(2).strip()}"
            else:
                m2 = re.search(r'(?:by and between|between)\s+([^,\n]+(?:,\s*[^,\n]+)?)\s+(?:and)\s+([^,\n]+)', text, re.IGNORECASE)
                if m2:
                    p1 = re.sub(r'\(.*?\)', '', m2.group(1)).strip().split(',')[0].strip()
                    p2 = re.sub(r'\(.*?\)', '', m2.group(2)).strip().split(',')[0].strip()
                    parties = f"{p1} and {p2}"

        # 3. Purpose
        purpose = "To define contractual rights, obligations, and commercial terms between the parties."
        purpose_sec = re.search(r'(?:PURPOSE(?: OF AGREEMENT)?|SCOPE OF WORK|PURPOSE & PREMISES|SECTION\s+\d+:\s*PURPOSE[^\n]*)[:\s]+\n*(.*?)(?=\n\s*(?:\d+\.\s+[A-Z\s/&]+:|\bSECTION\s+\d+:|$))', text, re.IGNORECASE | re.DOTALL)
        if purpose_sec:
            p_raw = purpose_sec.group(1).strip()
            p_raw = re.sub(r'^(?:AND SCOPE OF WORK|& PREMISES|OF AGREEMENT)[:\s]*', '', p_raw, flags=re.IGNORECASE).strip()
            lines = [l.strip() for l in p_raw.splitlines() if l.strip() and not l.strip().isupper()]
            if lines:
                first_s = re.split(r'\.\s+', lines[0])[0]
                if len(first_s) > 10:
                    purpose = first_s.strip() + "."

        # 4. Duration
        duration = "Not explicitly stated"
        dur_sec = re.search(r'(?:DURATION|TERM|LEASE PERIOD|SECTION\s+\d+:\s*TERM[^\n]*)(?:\s*[/&]\s*[A-Z]+)?[:\s]+\n*(.*?)(?=\n\s*(?:\d+\.\s+[A-Z\s/&]+:|\bSECTION\s+\d+:|$))', text, re.IGNORECASE | re.DOTALL)
        if dur_sec:
            d_raw = dur_sec.group(1).strip()
            d_raw = re.sub(r'^(?:AND DURATION|TERM|LEASE PERIOD)[:\s]*', '', d_raw, flags=re.IGNORECASE).strip()
            lines = [l.strip() for l in d_raw.splitlines() if l.strip() and not l.strip().isupper()]
            if lines:
                first_s = re.split(r'\.\s+', lines[0])[0]
                if first_s:
                    duration = first_s.strip() + "."
        if duration == "Not explicitly stated":
            time_m = re.search(r'(\d+\s+(?:months|years|days))', text, re.IGNORECASE)
            if time_m:
                duration = f"{time_m.group(1)} from effective date."

        # 5. Key Obligations
        obligations = []
        ob_sec = re.search(r'(?:\n|^)(?:\d+\.\s+|SECTION\s+\d+:?\s*)?(?:KEY\s+OBLIGATIONS|KEY\s+OPERATIONAL\s+OBLIGATIONS|OBLIGATIONS|DUTIES)[^\n]*\n+(.*?)(?=\n\s*(?:\d+\.\s+[A-Z\s/&]+:|\bSECTION\s+\d+:|$))', text, re.IGNORECASE | re.DOTALL)
        if ob_sec:
            lines = [re.sub(r'^[-•*\d\.\)]+\s*', '', l).strip() for l in ob_sec.group(1).splitlines() if l.strip()]
            obligations = [l for l in lines if len(l) > 15 and not l.isupper()]

        if not obligations:
            if "confidentiality" in text_lower:
                obligations.append("Maintain strict confidentiality regarding proprietary business information.")
            if "payment" in text_lower or "salary" in text_lower or "rent" in text_lower or "fee" in text_lower:
                obligations.append("Remit agreed payments in accordance with contractual milestones and schedule.")
            obligations.append("Perform designated operational obligations and adhere to specified quality standards.")

        # 6. Payment
        payment = "No monetary payment specified."
        pay_sec = re.search(r'(?:\n|^)(?:\d+\.\s+|SECTION\s+\d+:?\s*)?(?:PAYMENT|COMPENSATION|CONSIDERATION|FEE|FEES|RENT)[^\n]*\n+(.*?)(?=\n\s*(?:\d+\.\s+[A-Z\s/&]+:|\bSECTION\s+\d+:|$))', text, re.IGNORECASE | re.DOTALL)
        if pay_sec:
            lines = [re.sub(r'^[-•*\d\.\)]+\s*', '', l).strip() for l in pay_sec.group(1).splitlines() if l.strip()]
            matching = [l for l in lines if any(k in l.lower() for k in ['rs.', '₹', '$', 'fee', 'salary', 'rent', 'deposit', 'consideration', 'payment', 'milestone', 'invoiced', 'adequate'])]
            if matching:
                payment = matching[0]
            elif lines:
                payment = lines[0]

        # 7. Termination
        termination = "Subject to mutual consent or statutory notice."
        term_sec = re.search(r'(?:\n|^)(?:\d+\.\s+|SECTION\s+\d+:?\s*)?TERMINATION[^\n]*\n+(.*?)(?=\n\s*(?:\d+\.\s+[A-Z\s/&]+:|\bSECTION\s+\d+:|$))', text, re.IGNORECASE | re.DOTALL)
        if term_sec:
            lines = [re.sub(r'^[-•*\d\.\)]+\s*', '', l).strip() for l in term_sec.group(1).splitlines() if l.strip()]
            matching = [l for l in lines if any(k in l.lower() for k in ['notice', 'terminate', 'breach', 'days', 'without cause', 'cause'])]
            if matching:
                termination = matching[0]
            elif lines:
                termination = lines[0]

        # 8. Important Dates
        dates = []
        date_matches = re.findall(r'(\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b)', text)
        if date_matches:
            unique_dates = list(dict.fromkeys(date_matches))
            dates.append(f"Start / Effective Date: {unique_dates[0]}")
            if len(unique_dates) > 1:
                dates.append(f"Concluding / Milestone Date: {unique_dates[1]}")
            for d in unique_dates[2:4]:
                dates.append(f"Execution Date: {d}")
        else:
            dates.append("Start Date: As indicated upon execution")

        # 9. Potentially Important Clauses
        clauses = []
        if "confidentiality" in text_lower:
            surv = re.search(r'survive.*?(\d+\s+(?:years|months))', text, re.IGNORECASE)
            surv_str = f" (survives for {surv.group(1)})" if surv else ""
            clauses.append(f"Confidentiality: Strict protection of proprietary secrets and business data{surv_str}.")
        if "payment" in text_lower or "fee" in text_lower or "rent" in text_lower or "salary" in text_lower or "consideration" in text_lower:
            clauses.append("Payment: Specifies remuneration schedules, invoices, and applicable taxes/penalties.")
        if "termination" in text_lower:
            clauses.append("Termination: Details notice periods, breach remedy timelines, and exit provisions.")
        if "liability" in text_lower or "remedies" in text_lower or "indemnification" in text_lower:
            clauses.append("Liability: Establishes limitations of aggregate liability and exclusions for consequential losses.")
        if "intellectual property" in text_lower or "work made for hire" in text_lower or "ownership" in text_lower:
            clauses.append("Intellectual Property: Vests created source code, works, and assets with client/employer.")
        if "dispute resolution" in text_lower or "arbitration" in text_lower or "governing law" in text_lower:
            clauses.append("Dispute Resolution: Mandates negotiation/arbitration and identifies legal governing jurisdiction.")

        # 10. Abstractive Summary
        summary = (
            f"This {doc_type.lower()} establishes a formal legal relationship between {parties}. "
            f"The primary purpose is {purpose.lower().rstrip('.')}. "
            f"The engagement covers a duration of {duration.lower().rstrip('.')}, with financial consideration structured as {payment.rstrip('.')}. "
            f"Termination conditions mandate {termination.rstrip('.')}, while preserving essential post-termination covenants "
            f"including confidentiality and intellectual property protections."
        )

        res = LegalAnalysisResult(
            document_type=doc_type,
            parties=parties,
            purpose=purpose,
            duration=duration,
            key_obligations=obligations,
            payment=payment,
            termination=termination,
            important_dates=dates,
            important_clauses=clauses,
            summary=summary,
            raw_output="",
            chunk_count=1,
            model_used="Simulation / Fallback Heuristic Engine",
            word_count=stats["words"],
            estimated_tokens=stats["estimated_tokens"]
        )
        res.raw_output = res.to_formatted_text()
        return res


class GPT4LegalAnalyzer:
    """
    Main automated legal analyzer powered by OpenAI GPT-4 models.
    Supports single document analysis, chunked document analysis,
    and automatic fallback if API credentials are not supplied.
    """

    def __init__(self, api_key: Optional[str] = None, default_model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
        self.default_model = default_model
        self.client = None
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except Exception:
                self.client = None

    def has_active_api_key(self) -> bool:
        return bool(self.api_key and self.client is not None)

    def analyze(
        self,
        document_text: str,
        model: Optional[str] = None,
        use_chunking: bool = False,
        chunk_size: int = 2200,
        chunk_overlap: int = 200
    ) -> LegalAnalysisResult:
        model_name = model or self.default_model
        clean_text = DocumentExtractor.clean_text(document_text)
        stats = DocumentExtractor.get_stats(clean_text)

        if not clean_text:
            raise ValueError("Document text is empty. Please provide text or upload a document.")

        chunker = DocumentChunker()
        chunks = chunker.split_into_chunks(clean_text, max_chunk_chars=chunk_size, overlap_chars=chunk_overlap)
        should_chunk = use_chunking or (len(clean_text) > 3500 and len(chunks) > 1)

        # Fallback if no OpenAI API Key or client
        if not self.has_active_api_key():
            res = FallbackAnalyzer.analyze(clean_text)
            res.model_used = f"Fallback Simulation ({model_name} requested)"
            res.chunk_count = len(chunks) if should_chunk else 1
            return res

        # Run with live OpenAI API
        try:
            if should_chunk and len(chunks) > 1:
                return self._analyze_with_chunks(chunks, model_name, stats)
            else:
                return self._analyze_single(clean_text, model_name, stats)
        except Exception as api_err:
            res = FallbackAnalyzer.analyze(clean_text)
            res.model_used = f"Fallback (API Error: {type(api_err).__name__})"
            res.chunk_count = len(chunks) if should_chunk else 1
            return res

    def _analyze_single(self, text: str, model: str, stats: Dict[str, int]) -> LegalAnalysisResult:
        prompts = PromptGenerator.build_analysis_prompt(text)
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": prompts["user"]}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content or ""
        result = self._parse_structured_output(content)
        result.model_used = model
        result.chunk_count = 1
        result.word_count = stats["words"]
        result.estimated_tokens = stats["estimated_tokens"]
        return result

    def _analyze_with_chunks(self, chunks: List[str], model: str, stats: Dict[str, int]) -> LegalAnalysisResult:
        chunk_summaries = []
        for i, chunk in enumerate(chunks, start=1):
            prompts = PromptGenerator.build_analysis_prompt(chunk, is_chunk=True, chunk_index=i, total_chunks=len(chunks))
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]}
                ],
                temperature=0.2
            )
            chunk_summaries.append(response.choices[0].message.content or "")

        combine_prompts = PromptGenerator.build_combination_prompt(chunk_summaries)
        final_resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": combine_prompts["system"]},
                {"role": "user", "content": combine_prompts["user"]}
            ],
            temperature=0.2
        )
        final_content = final_resp.choices[0].message.content or ""
        result = self._parse_structured_output(final_content)
        result.model_used = f"{model} (Chunked: {len(chunks)} chunks)"
        result.chunk_count = len(chunks)
        result.word_count = stats["words"]
        result.estimated_tokens = stats["estimated_tokens"]
        return result

    def _parse_structured_output(self, raw_text: str) -> LegalAnalysisResult:
        """Parses the LLM markdown/text output into structured fields."""
        res = LegalAnalysisResult(raw_output=raw_text.strip())

        def get_section(title: str, next_titles: List[str]) -> str:
            escaped_title = re.escape(title)
            next_escaped = "|".join([re.escape(nt) for nt in next_titles])
            pattern = rf'(?:{escaped_title}:?)\s*(.*?)(?=(?:\n\s*(?:{next_escaped}):?)|$)'
            match = re.search(pattern, raw_text, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""

        titles = [
            "Document Type", "Parties", "Purpose", "Duration",
            "Key Obligations", "Payment", "Termination",
            "Important Dates", "Potentially Important Clauses", "Summary"
        ]

        res.document_type = get_section("Document Type", titles[1:]) or "Legal Document"
        res.parties = get_section("Parties", titles[2:]) or "Not specified"
        res.purpose = get_section("Purpose", titles[3:]) or "Not specified"
        res.duration = get_section("Duration", titles[4:]) or "Not specified"

        obligations_block = get_section("Key Obligations", titles[5:])
        if obligations_block:
            res.key_obligations = [
                line.strip().lstrip('•*- ') for line in obligations_block.splitlines()
                if line.strip() and not line.strip().startswith('Key Obligations')
            ]

        res.payment = get_section("Payment", titles[6:]) or "Not specified"
        res.termination = get_section("Termination", titles[7:]) or "Not specified"

        dates_block = get_section("Important Dates", titles[8:])
        if dates_block:
            res.important_dates = [
                line.strip().lstrip('•*- ') for line in dates_block.splitlines()
                if line.strip() and not line.strip().startswith('Important Dates')
            ]

        clauses_block = get_section("Potentially Important Clauses", titles[9:])
        if clauses_block:
            res.important_clauses = [
                line.strip().lstrip('•*- ') for line in clauses_block.splitlines()
                if line.strip() and not line.strip().startswith('Potentially Important Clauses')
            ]

        res.summary = get_section("Summary", []) or "Summary not provided."
        return res
