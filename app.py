"""
app.py
Interactive Streamlit Web Application for
24CS2506 - Building Applications using GPT-4
Experiment 5: Automated Legal Document Analyzer using GPT-4
"""

import os
import json
import streamlit as st
from legal_analyzer import (
    DocumentExtractor,
    DocumentChunker,
    GPT4LegalAnalyzer,
    PromptGenerator,
    LegalAnalysisResult
)

# Page configuration
st.set_page_config(
    page_title="Automated Legal Document Analyzer | GPT-4",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .clause-card {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .clause-title {
        font-weight: 600;
        color: #1E40AF;
        margin-bottom: 4px;
    }
    .status-badge-live {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-sim {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "Employment Agreement"


# Sidebar Configuration
st.sidebar.title("⚖️ Legal Analyzer")
st.sidebar.caption("Experiment 5: GPT-4 Application")

# API Key handling
api_key_env = os.getenv("OPENAI_API_KEY", "").strip()
user_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=api_key_env,
    type="password",
    help="Enter OpenAI API Key. If empty, the system uses the realistic heuristic simulation engine."
)

effective_api_key = user_api_key.strip() if user_api_key else api_key_env

if effective_api_key:
    st.sidebar.markdown('<span class="status-badge-live">🟢 Live OpenAI API Connected</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-badge-sim">🟡 Simulation Engine Active (No Key)</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# Model Selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
    index=0
)

# Chunking Options
st.sidebar.subheader("Document Chunking (Sec. 9)")
enable_chunking = st.sidebar.checkbox(
    "Enable Chunking",
    value=False,
    help="Splits lengthy documents into smaller sections, processes each section, and combines results."
)

chunk_size = 2200
chunk_overlap = 200
if enable_chunking:
    chunk_size = st.sidebar.slider("Chunk Size (characters)", 1000, 4000, 2200, step=200)
    chunk_overlap = st.sidebar.slider("Overlap (characters)", 50, 400, 200, step=50)

st.sidebar.markdown("---")
st.sidebar.info("""
**Lab 5 Information Extraction Targets:**
- Parties & Purpose
- Key Obligations & Rights
- Payment Terms & Currency
- Duration & Important Dates
- Termination Conditions
- Key Clauses (Confidentiality, Liability, IP, Dispute Resolution)
- Abstractive Summary
""")


# Main App Layout
st.markdown('<div class="main-header">Automated Legal Document Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">24CS2506 – Building Applications using GPT-4 | Experiment 5</div>', unsafe_allow_html=True)

# Sample document dictionary
SAMPLES = {
    "Employment Agreement": "sample_documents/employment_agreement.txt",
    "Mutual Non-Disclosure Agreement (NDA)": "sample_documents/non_disclosure_agreement.txt",
    "Master Software Services Agreement": "sample_documents/service_agreement.txt",
    "Commercial Lease & Rental Agreement": "sample_documents/rental_agreement.txt",
    "Long Commercial Contract (Multi-section)": "sample_documents/long_commercial_contract.txt"
}

# Input Tabs
input_tab1, input_tab2, input_tab3 = st.tabs([
    "📂 Sample Legal Documents",
    "📤 Upload Document (.txt, .pdf)",
    "✍️ Direct Text Paste"
])

with input_tab1:
    col_samp1, col_samp2 = st.columns([3, 1])
    with col_samp1:
        chosen_sample = st.selectbox("Choose a pre-packaged legal agreement:", list(SAMPLES.keys()))
    with col_samp2:
        load_btn = st.button("Load Sample Document", use_container_width=True)

    if load_btn or (st.session_state.selected_sample != chosen_sample and not st.session_state.document_text):
        st.session_state.selected_sample = chosen_sample
        sample_path = SAMPLES[chosen_sample]
        try:
            st.session_state.document_text = DocumentExtractor.extract_from_file(sample_path)
            st.session_state.analysis_result = None
            st.success(f"Loaded '{chosen_sample}' ({len(st.session_state.document_text)} characters)")
        except Exception as e:
            st.error(f"Failed to load sample: {e}")

with input_tab2:
    uploaded_file = st.file_uploader("Upload legal agreement file", type=["txt", "pdf"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".pdf"):
                import pypdf
                reader = pypdf.PdfReader(uploaded_file)
                pdf_text = ""
                for page in reader.pages:
                    p_txt = page.extract_text()
                    if p_txt:
                        pdf_text += p_txt + "\n"
                st.session_state.document_text = DocumentExtractor.clean_text(pdf_text)
            else:
                raw_bytes = uploaded_file.read()
                try:
                    raw_str = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    raw_str = raw_bytes.decode("latin-1")
                st.session_state.document_text = DocumentExtractor.clean_text(raw_str)
            st.session_state.analysis_result = None
            st.success(f"File '{uploaded_file.name}' loaded successfully!")
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")

with input_tab3:
    pasted_text = st.text_area(
        "Paste contract or agreement text here:",
        value=st.session_state.document_text,
        height=240,
        key="custom_text_input"
    )
    if st.button("Update Document Text"):
        st.session_state.document_text = DocumentExtractor.clean_text(pasted_text)
        st.session_state.analysis_result = None
        st.success("Document updated!")

# Display current document summary
current_text = st.session_state.document_text
if current_text:
    stats = DocumentExtractor.get_stats(current_text)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Characters", f"{stats['characters']:,}")
    with col_m2:
        st.metric("Total Words", f"{stats['words']:,}")
    with col_m3:
        st.metric("Estimated Tokens", f"~{stats['estimated_tokens']:,}")
    with col_m4:
        st.metric("Total Lines", f"{stats['lines']:,}")

    with st.expander("📄 View Loaded Document Text", expanded=False):
        st.text_area("Contract Text", current_text, height=220, disabled=True)

    # Action button
    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze Legal Document with GPT-4", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("Processing document through legal extraction pipeline..."):
            analyzer = GPT4LegalAnalyzer(
                api_key=effective_api_key,
                default_model=model_choice
            )
            try:
                res = analyzer.analyze(
                    document_text=current_text,
                    model=model_choice,
                    use_chunking=enable_chunking,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                st.session_state.analysis_result = res
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Display Analysis Results
result = st.session_state.analysis_result
if result:
    st.markdown("### 📋 Analysis Results")

    # Metrics row
    r_c1, r_c2, r_c3, r_c4 = st.columns(4)
    with r_c1:
        st.metric("Document Type", result.document_type)
    with r_c2:
        st.metric("Processing Engine", result.model_used)
    with r_c3:
        st.metric("Chunks Processed", result.chunk_count)
    with r_c4:
        st.metric("Obligations Found", len(result.key_obligations))

    # Results tabs
    res_tab1, res_tab2, res_tab3, res_tab4, res_tab5 = st.tabs([
        "📜 Standard Output Format",
        "⚖️ Clauses & Obligations",
        "🎯 Extractive vs Abstractive",
        "🧩 Chunking Inspector",
        "💾 Export & Download"
    ])

    with res_tab1:
        st.caption("Matches Experiment 5 Recommended Output Format:")
        formatted_txt = result.to_formatted_text()
        st.code(formatted_txt, language="text")

    with res_tab2:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### 👥 Parties & Purpose")
            st.markdown(f"""
            <div class="clause-card">
                <div class="clause-title">Parties Involved</div>
                <div>{result.parties}</div>
            </div>
            <div class="clause-card">
                <div class="clause-title">Purpose of Agreement</div>
                <div>{result.purpose}</div>
            </div>
            <div class="clause-card">
                <div class="clause-title">Duration / Term</div>
                <div>{result.duration}</div>
            </div>
            <div class="clause-card">
                <div class="clause-title">Payment Terms</div>
                <div>{result.payment}</div>
            </div>
            <div class="clause-card">
                <div class="clause-title">Termination Conditions</div>
                <div>{result.termination}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_c2:
            st.markdown("#### 🛡️ Key Clauses & Obligations")
            st.markdown("**Key Obligations:**")
            if result.key_obligations:
                for ob in result.key_obligations:
                    st.markdown(f"- ✅ **{ob.lstrip('•*- ')}**")
            else:
                st.write("No specific obligations listed.")

            st.markdown("**Potentially Important Clauses (Sec. 7):**")
            if result.important_clauses:
                for cl in result.important_clauses:
                    st.markdown(f"""
                    <div class="clause-card">
                        <div>{cl}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("General contractual provisions apply.")

    with res_tab3:
        st.markdown("#### Section 8: Extractive vs Abstractive Summarization")
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.markdown("**Original Document Sample (Extractive Context):**")
            sample_preview = "\n\n".join(current_text.split("\n\n")[:3])
            st.info(f"\"{sample_preview}\"")
            st.caption("Extractive summarization selects verbatim sentences directly from the raw document.")
        with col_sum2:
            st.markdown("**GPT-4 Abstractive Summary:**")
            st.success(result.summary)
            st.caption("Abstractive summarization synthesizes the fundamental legal rights, obligations, and implications in concise original language.")

    with res_tab4:
        st.markdown("#### Section 9: Chunking Long Documents")
        chunker = DocumentChunker()
        chunks = chunker.split_into_chunks(current_text, max_chunk_chars=chunk_size, overlap_chars=chunk_overlap)
        st.write(f"The document was divided into **{len(chunks)} chunk(s)** (Chunk Size: {chunk_size} chars, Overlap: {chunk_overlap} chars).")

        for idx, ch in enumerate(chunks, start=1):
            with st.expander(f"📦 Chunk {idx} ({len(ch)} chars, ~{len(ch)//4} tokens)"):
                st.text(ch)

    with res_tab5:
        st.markdown("#### Export Analysis Results")
        col_d1, col_d2, col_d3 = st.columns(3)
        formatted_txt = result.to_formatted_text()
        json_data = json.dumps(result.to_dict(), indent=2)
        md_data = f"# Legal Document Analysis\n\n```text\n{formatted_txt}\n```\n"

        with col_d1:
            st.download_button(
                "📥 Download as Text (.txt)",
                data=formatted_txt,
                file_name=f"legal_analysis_{result.document_type.lower().replace(' ', '_')[:20]}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_d2:
            st.download_button(
                "📥 Download as JSON (.json)",
                data=json_data,
                file_name=f"legal_analysis_{result.document_type.lower().replace(' ', '_')[:20]}.json",
                mime="application/json",
                use_container_width=True
            )
        with col_d3:
            st.download_button(
                "📥 Download as Markdown (.md)",
                data=md_data,
                file_name=f"legal_analysis_{result.document_type.lower().replace(' ', '_')[:20]}.md",
                mime="text/markdown",
                use_container_width=True
            )
else:
    if not current_text:
        st.info("👋 Select a sample document or upload an agreement above to begin analysis.")
