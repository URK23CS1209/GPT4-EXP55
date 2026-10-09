"""
test_analyzer.py
Unit and Integration Tests for Experiment 5:
Automated Legal Document Analyzer using GPT-4
"""

import os
import pytest
from legal_analyzer import (
    DocumentExtractor,
    DocumentChunker,
    PromptGenerator,
    FallbackAnalyzer,
    GPT4LegalAnalyzer,
    LegalAnalysisResult
)


class TestDocumentExtractor:
    def test_clean_text(self):
        dirty = "  Clause 1.\r\n\r\n\r\n\r\nPayment shall be \u00a0 made.  "
        cleaned = DocumentExtractor.clean_text(dirty)
        assert "\r" not in cleaned
        assert "\u00a0" not in cleaned
        assert "Payment shall be   made." in cleaned
        assert cleaned.startswith("Clause 1.")

    def test_get_stats(self):
        text = "This is a sample legal text containing ten words total."
        stats = DocumentExtractor.get_stats(text)
        assert stats["words"] == 10
        assert stats["characters"] == len(text)
        assert stats["estimated_tokens"] == len(text) // 4
        assert stats["lines"] == 1

    def test_extract_from_file_existing(self):
        path = "sample_documents/employment_agreement.txt"
        assert os.path.exists(path)
        content = DocumentExtractor.extract_from_file(path)
        assert len(content) > 500
        assert "EMPLOYMENT AGREEMENT" in content

    def test_extract_from_file_missing(self):
        with pytest.raises(FileNotFoundError):
            DocumentExtractor.extract_from_file("sample_documents/non_existent.txt")


class TestDocumentChunker:
    def test_short_document_no_chunking(self):
        short = "This is a short contract text with only few words."
        chunks = DocumentChunker.split_into_chunks(short, max_chunk_chars=1000)
        assert len(chunks) == 1
        assert chunks[0] == short

    def test_empty_document(self):
        chunks = DocumentChunker.split_into_chunks("")
        assert chunks == []

    def test_long_document_chunking(self):
        path = "sample_documents/long_commercial_contract.txt"
        text = DocumentExtractor.extract_from_file(path)
        chunks = DocumentChunker.split_into_chunks(text, max_chunk_chars=1500, overlap_chars=100)
        assert len(chunks) > 1
        # Check that all chunks are non-empty
        for ch in chunks:
            assert len(ch) > 0


class TestPromptGenerator:
    def test_prompt_constraints(self):
        sample = "Agreement between Party A and Party B."
        prompts = PromptGenerator.build_analysis_prompt(sample)
        sys_prompt = prompts["system"]
        user_prompt = prompts["user"]

        assert "legal document analysis assistant" in sys_prompt.lower()
        assert "no hallucinations" in sys_prompt.lower() or "do not invent facts" in sys_prompt.lower()
        assert "LEGAL DOCUMENT ANALYSIS" in user_prompt
        assert "Document Type:" in user_prompt
        assert "Key Obligations:" in user_prompt
        assert "Potentially Important Clauses:" in user_prompt

    def test_combination_prompt(self):
        chunk_summaries = ["Summary chunk 1", "Summary chunk 2"]
        prompts = PromptGenerator.build_combination_prompt(chunk_summaries)
        assert "synthesis assistant" in prompts["system"].lower()
        assert "LEGAL DOCUMENT ANALYSIS" in prompts["user"]


class TestFallbackAnalyzer:
    def test_employment_agreement(self):
        path = "sample_documents/employment_agreement.txt"
        text = DocumentExtractor.extract_from_file(path)
        result = FallbackAnalyzer.analyze(text)

        assert result.document_type == "Employment Agreement"
        assert "Apex Innovations" in result.parties
        assert "Rahul Sharma" in result.parties
        assert "85,000" in result.payment
        assert "30" in result.termination
        assert len(result.key_obligations) >= 2
        assert len(result.important_clauses) >= 4
        assert len(result.summary) > 50

    def test_nda(self):
        path = "sample_documents/non_disclosure_agreement.txt"
        text = DocumentExtractor.extract_from_file(path)
        result = FallbackAnalyzer.analyze(text)

        assert "Non-Disclosure" in result.document_type or "NDA" in result.document_type
        assert "TechCore" in result.parties
        assert "DataVibe" in result.parties
        assert len(result.key_obligations) >= 1
        assert "14" in result.termination or "days" in result.termination

    def test_service_agreement(self):
        path = "sample_documents/service_agreement.txt"
        text = DocumentExtractor.extract_from_file(path)
        result = FallbackAnalyzer.analyze(text)

        assert "Service" in result.document_type
        assert "Alpha" in result.parties
        assert "CloudScale" in result.parties
        assert "4,50,000" in result.payment
        assert "15" in result.termination

    def test_rental_agreement(self):
        path = "sample_documents/rental_agreement.txt"
        text = DocumentExtractor.extract_from_file(path)
        result = FallbackAnalyzer.analyze(text)

        assert "Rental" in result.document_type or "Lease" in result.document_type
        assert "Vikram Oberoi" in result.parties
        assert "NextGen Analytics" in result.parties
        assert "65,000" in result.payment

    def test_formatted_text_output(self):
        path = "sample_documents/employment_agreement.txt"
        text = DocumentExtractor.extract_from_file(path)
        result = FallbackAnalyzer.analyze(text)
        formatted = result.to_formatted_text()

        assert "LEGAL DOCUMENT ANALYSIS" in formatted
        assert "--------------------------------" in formatted
        assert "Document Type:" in formatted
        assert "Parties:" in formatted
        assert "Purpose:" in formatted
        assert "Duration:" in formatted
        assert "Key Obligations:" in formatted
        assert "Payment:" in formatted
        assert "Termination:" in formatted
        assert "Important Dates:" in formatted
        assert "Potentially Important Clauses:" in formatted
        assert "Summary:" in formatted


class TestGPT4LegalAnalyzer:
    def test_analyzer_fallback_without_api_key(self):
        # Empty API key should default to Fallback Simulation without crashing
        analyzer = GPT4LegalAnalyzer(api_key="")
        assert not analyzer.has_active_api_key()

        text = "This is a simple contract between Acme Corp and Beta LLC for software consulting."
        res = analyzer.analyze(text)
        assert isinstance(res, LegalAnalysisResult)
        assert "Fallback Simulation" in res.model_used
        assert res.document_type != ""
        assert len(res.summary) > 10

    def test_chunked_analysis_workflow(self):
        path = "sample_documents/long_commercial_contract.txt"
        text = DocumentExtractor.extract_from_file(path)
        analyzer = GPT4LegalAnalyzer(api_key="")
        res = analyzer.analyze(text, use_chunking=True, chunk_size=1500)
        assert res.chunk_count > 1
        assert "Horizon Global Enterprises" in res.parties
        assert "Matrix Cloud Technologies" in res.parties
