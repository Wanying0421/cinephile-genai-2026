import os
import sys
import time
import streamlit as st
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    import level0
except Exception as e:
    level0 = None

try:
    import level1
except Exception as e:
    level1 = None

st.set_page_config(
    page_title="Cinephile GenAI - RAG Evaluation UI",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Cinephile GenAI RAG Evaluator")
st.markdown("Test and evaluate **Level 0** and **Level 1** RAG pipelines.")

selected_level = st.sidebar.selectbox(
    "Select RAG Level",
    ["Level 0 (Baseline)", "Level 1 (Advanced RAG)", "Level 2 (Placeholder)", "Level 3 (Placeholder)"]
)

st.sidebar.markdown("---")
st.sidebar.info("Member C Evaluation & UI Dashboard")

query = st.text_input("Enter your movie question:", placeholder="e.g., What is the summary of the movie Inception?")

if st.button("Run Query", type="primary"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        answer = ""
        sources = []
        tokens = 0
        latency = 0.0

        if selected_level == "Level 0 (Baseline)":
            if level0 and hasattr(level0, "run"):
                try:
                    res = level0.run(query)
                    if isinstance(res, dict):
                        answer = res.get("answer", "")
                        tokens = res.get("prompt_tokens", 0) + res.get("completion_tokens", 0)
                        latency = res.get("latency_s", 0.0)
                    else:
                        answer = str(res)
                except Exception as e:
                    answer = f"Error running Level 0: {e}"
            else:
                answer = "Level 0 module (`level0.py`) could not be loaded. Check dependencies."

        elif selected_level == "Level 1 (Advanced RAG)":
            if level1 and hasattr(level1, "rag_answer"):
                try:
                    res = level1.rag_answer(query)
                    if isinstance(res, dict):
                        answer = res.get("answer", "")
                        sources = res.get("citations", [])
                        usage = res.get("usage", {})
                        tokens = usage.get("total_tokens", 0)
                        latency = res.get("latency", 0.0)
                    else:
                        answer = str(res)
                except Exception as e:
                    answer = f"Error running Level 1: {e}"
            else:
                answer = "Level 1 module (`level1.py`) could not be loaded. Check Faiss/dependencies."

        else:
            answer = f"**{selected_level}** is not implemented yet."

        st.markdown("### 📝 Generated Answer")
        st.write(answer if answer else "No response generated.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📚 Cited Sources")
            if sources:
                for src in sources:
                    st.code(str(src))
            else:
                st.markdown("_No sources cited or available._")

        with col2:
            st.markdown("### 📊 Trace Information")
            st.metric(label="Latency", value=f"{latency:.3f} seconds")
            st.metric(label="Estimated Tokens", value=tokens)