# blood_report_analyzer_groq.py
# RAG chatbot for blood reports using Groq + LangChain + Streamlit
# Suitable for local use and Streamlit Community Cloud

import streamlit as st
import pandas as pd
from io import StringIO
import time
from datetime import datetime

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

# Groq integration
from langchain_groq import ChatGroq

st.set_page_config(page_title="Blood Report Analyzer • Groq", layout="wide")

# ────────────────────────────────────────────────
# Embeddings
# ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = load_embeddings()

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None

# ────────────────────────────────────────────────
# Sidebar - Groq API Key (secure input)
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Groq API Key (required)")

    groq_api_key = st.text_input(
        label="Enter your Groq API key",
        type="password",
        placeholder="gsk_...",
        value="",
        key="groq_api_key_widget",
        help="Get your key → https://console.groq.com/keys\nKey is only used in this session."
    )

    if not groq_api_key.strip():
        st.warning("Groq API key is required to use the chat.")
        st.info("Sign up / get key: https://console.groq.com/keys")
        st.stop()

    st.markdown("---")
    st.caption("Model: **llama-3.3-70b-versatile** via Groq")

# ────────────────────────────────────────────────
# Main UI
# ────────────────────────────────────────────────
st.title("🩸 Blood Report Analyzer – Groq Edition")
st.caption("Paste table → Edit → Process → Ask questions • Powered by Groq")

tab1, tab2 = st.tabs(["📊 Paste & Edit Table", "ℹ️ How to use"])

with tab1:
    st.markdown(
        "Paste your blood report table (from PDF, lab portal, Excel, WhatsApp, etc.)\n"
        "Best results when columns are separated by **comma**, **tab** or **spaces**."
    )

    raw_paste = st.text_area(
        "1. Paste raw tabular text here",
        height=220,
        value="""Test,Result,Unit,Reference Range,Flag
Hemoglobin,12.4,g/dL,13.0 - 17.0,L
WBC,8.2,10^3/µL,4.0 - 11.0,
Glucose (Fasting),102,mg/dL,70 - 99,H
Creatinine,1.1,mg/dL,0.6 - 1.2,
ALT,45,U/L,7 - 56,
Total Cholesterol,210,mg/dL,<200,H""",
        help="Best if columns are separated by comma or tab"
    )

    if st.button("2. Parse pasted text → Show editable table", type="primary"):
        if raw_paste.strip():
            try:
                df = pd.read_csv(StringIO(raw_paste), sep=None, engine="python", on_bad_lines="skip")
                df = df.dropna(how="all")
                st.session_state.df = df
                st.success(f"Table parsed! {len(df)} rows detected.")
            except Exception as e:
                st.error(f"Parsing failed: {str(e)}")
        else:
            st.warning("Please paste some table text first.")

    if st.session_state.df is not None:
        st.markdown("3. Edit values directly in the table below")
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=False,
            column_config={
                "Test": st.column_config.TextColumn("Test name", required=True),
                "Result": st.column_config.NumberColumn("Result", min_value=0.0, step=0.01),
                "Unit": st.column_config.TextColumn("Unit"),
                "Reference Range": st.column_config.TextColumn("Reference range"),
                "Flag": st.column_config.SelectboxColumn(
                    "Flag",
                    options=["", "H", "L", "H*", "L*", "Critical", "Abnormal"],
                    required=False
                ),
            }
        )

        if st.button("4. Process edited table → Ready for questions", type="primary"):
            with st.spinner("Building vector index..."):
                lines = ["Test | Result | Unit | Reference Range | Flag"]
                for _, row in edited_df.iterrows():
                    row_str = " | ".join(str(val) for val in row if pd.notna(val) and str(val).strip())
                    if row_str.strip():
                        lines.append(row_str)

                full_text = "\n".join(lines)

                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = splitter.split_text(full_text)
                docs = [Document(page_content=ch) for ch in chunks]

                vectorstore = FAISS.from_documents(docs, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

                prompt_template = """You are a careful lab report assistant.
Use ONLY the information from the report table excerpts below.
If a value is missing or normal → say "not found in report" or "within normal range".
Never diagnose diseases. Only report values, flags, ranges.

Report table excerpts:
{context}

Question: {input}

Answer (concise, factual, include unit/range/flag when available):"""
                prompt = ChatPromptTemplate.from_template(prompt_template)

                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.15,
                    max_tokens=1200,
                    api_key=groq_api_key
                )

                qa_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, qa_chain)

                st.session_state.rag_chain = rag_chain

            st.success(f"Table processed! ({len(chunks)} chunks) → Ask questions now.")

    # ── Chat area ────────────────────────────────────────
    if st.session_state.rag_chain is not None:
        st.divider()
        st.markdown("### Ask questions about the current report")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if query := st.chat_input("Ask anything about the report (e.g. 'Is glucose high?')"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    try:
                        response = st.session_state.rag_chain.invoke({"input": query})
                        answer = response["answer"].strip()
                        st.markdown(answer)
                        elapsed = time.time() - start_time
                        st.caption(f"Answered in {elapsed:.1f} seconds")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        answer = f"Error: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": answer})

        # ── Download Q&A ────────────────────────────────────────────────
        if st.session_state.messages:
            st.markdown("---")

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            md_content = "# Blood Report Q&A\n"
            md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    md_content += f"**You:**\n{msg['content']}\n\n"
                else:
                    md_content += f"**Assistant:**\n{msg['content']}\n\n"
                    md_content += "---\n\n"

            st.download_button(
                label="📥 Download this Q&A conversation",
                data=md_content,
                file_name=f"blood_report_qa_{timestamp}.md",
                mime="text/markdown",
                help="Saves all questions and answers in nicely formatted markdown",
                use_container_width=False
            )

# ────────────────────────────────────────────────
# How to use tab
# ────────────────────────────────────────────────
with tab2:
    st.markdown("""
    ### How to use this tool

    1. Paste your blood report table text (from PDF, website, Excel, WhatsApp, etc.)
    2. Click **Parse** to see the editable table
    3. Correct any values or add missing rows if needed
    4. Click **Process** to create the question-answering index
    5. Ask questions in the chat box below
    6. Download the full Q&A conversation when finished

    **Note**: This is an educational/experimental tool.  
    Always consult a qualified doctor for medical interpretation.
    """)

st.caption("Made with Streamlit • LangChain • Groq • HuggingFace embeddings")
