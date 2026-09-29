"""Optional Streamlit chat UI.

Run:  streamlit run streamlit_app.py
"""

import streamlit as st

from src.graph import build_rag_graph, run

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")
st.title("Agentic AI eBook Chatbot")
st.caption("Answers come only from the Agentic AI eBook. Out-of-scope questions are refused.")


@st.cache_resource
def get_graph():
    return build_rag_graph()


if "history" not in st.session_state:
    st.session_state.history = []

chat_col, context_col = st.columns([3, 2])

with chat_col:
    for turn in st.session_state.history:
        st.chat_message("user").write(turn["query"])
        st.chat_message("assistant").write(turn["final_answer"])

    query = st.chat_input("Ask about the eBook...")
    if query:
        st.chat_message("user").write(query)
        try:
            with st.spinner("Retrieving and checking the answer..."):
                payload, chunks = run(get_graph(), query)
        except Exception as exc:
            st.error(f"The model service is busy or unavailable, please try again. ({exc})"[:300])
        else:
            st.chat_message("assistant").write(payload["final_answer"])
            st.session_state.history.append({**payload, "_chunks": chunks})

with context_col:
    if st.session_state.history:
        last = st.session_state.history[-1]
        payload = {k: v for k, v in last.items() if not k.startswith("_")}
        st.metric("Confidence score", f"{payload['confidence_score']:.2f}")
        st.subheader("Retrieved context")
        if not last["_chunks"]:
            st.caption("No relevant passages in the eBook for this question.")
        for i, chunk in enumerate(last["_chunks"], start=1):
            with st.expander(f"Chunk {i}: page {chunk['page_number']} "
                             f"(similarity {chunk['score']:.2f})"):
                st.write(chunk["text"])
        with st.expander("Raw JSON", expanded=True):
            st.json(payload)
