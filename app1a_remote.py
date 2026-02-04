# blood_report_analyzer_groq.py
# Version with Groq API - suitable for Streamlit Cloud / GitHub
import streamlit as st
import pandas as pd
from io import StringIO
import time
from datetime import datetime

# ────────────────────────────────────────────────
# LangChain & embeddings (cloud-friendly)
# ────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

# Groq client
from groq import Groq

st.set_page_config(page_title="Blood Report Analyzer • Groq", layout="wide")

# ────────────────────────────────────────────────
# Groq API Key handling
# ────────────────────────────────────────────────
def get_groq_client():
    # Priority: secrets > text input > environment
    api_key = None
    
    # 1. Try Streamlit secrets (recommended for cloud)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        pass
    
    # 2. Fallback to input field (for local testing)
    if not api_key:
        api_key = st.sidebar.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your key at: https://console.groq.com/keys"
        )
    
    if not api_key:
        st.warning("Please enter your Groq API key to continue.")
        st.stop()
    
    return Groq(api_key=api_key)


# ────────────────────────────────────────────────
# Embeddings (works everywhere - no local Ollama needed)
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
# UI
# ────────────────────────────────────────────────
st.title("🩸 Blood Report Analyzer – Groq Edition")
st.caption("Paste → Edit table → Process → Ask questions • Powered by Groq • Internet required")

tab1, tab2 = st.tabs(["📊 Paste & Edit Table", "ℹ️ How to use"])

with tab1:
    st.markdown(
        "Paste your blood report table (copy from PDF, lab portal, Excel, WhatsApp, etc.)\n"
        "Columns should be separated by **tabs**, **commas**, or **spaces**."
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
                # Convert table to text
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

                # Prompt
                prompt_template = """You are a careful lab report assistant.
Use ONLY the information from the report table excerpts below.
If a value is missing or normal → say "not found in report" or "within normal range".
Never diagnose diseases. Only report values, flags, ranges.

Report table excerpts:
{context}

Question: {input}

Answer (concise, factual, include unit/range/flag):"""
                prompt = ChatPromptTemplate.from_template(prompt_template)

                # Groq LLM wrapper (LangChain compatible)
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    model="llama-3.1-70b-versatile",   # or "mixtral-8x7b-32768", "gemma2-9b-it", etc.
                    temperature=0.15,
                    max_tokens=1200,
                    api_key=st.secrets.get("GROQ_API_KEY") or st.session_state.get("groq_api_key")
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
                    md_content += "---\n\n"  # separator after each assistant reply
            
            # Download button
            st.download_button(
                label="📥 Download this Q&A conversation",
                data=md_content,
                file_name=f"blood_report_qa_{timestamp}.md",
                mime="text/markdown",
                help="Saves all questions and answers in nicely formatted markdown",
                use_container_width=False
            )