"""Streamlit chat UI that talks to the FastAPI backend.  Run: streamlit run streamlit_app.py"""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REFUSAL_MESSAGE = "I cannot answer based on the provided document."

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="📘", layout="centered")
st.title("📘 Agentic AI eBook Chatbot")
st.caption(
    "Ask a question. Answers come only from the Agentic AI eBook, with page references. "
    "Scores show retrieval relevance (vector similarity), not the probability an answer is correct."
)


def call_api(query: str) -> dict:
    """Returns the API payload, or {'error': friendly message}."""
    try:
        resp = requests.post(f"{API_BASE_URL}/chat", json={"query": query}, timeout=120)
    except requests.ConnectionError:
        return {"error": f"Cannot reach the backend at {API_BASE_URL}. Is `uvicorn app:app` running?"}
    except requests.Timeout:
        return {"error": "The request timed out. Please try again."}
    except requests.RequestException:
        return {"error": "Unexpected network error while contacting the backend."}
    if resp.status_code == 200:
        return resp.json()
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "Please enter a valid question."
    return {"error": detail or f"The backend returned an error (HTTP {resp.status_code})."}


def render_assistant(payload: dict) -> None:
    answer = payload["final_answer"]
    unsupported = answer.strip() == REFUSAL_MESSAGE
    if unsupported:
        st.warning(f"**{REFUSAL_MESSAGE}**  \nThe eBook does not appear to contain enough information.")
    else:
        st.markdown(answer)
        st.progress(min(max(payload["confidence_score"], 0.0), 1.0),
                    text=f"Overall retrieval relevance: {payload['confidence_score']:.2f}")
        if payload["citations"]:
            st.markdown("**Sources:** " + ", ".join(
                f"{c['source']} (p. {c['page']})" for c in payload["citations"]))
    chunks = payload.get("retrieved_context", [])
    if chunks:
        label = "Closest passages (below relevance threshold)" if unsupported else "Retrieved context"
        with st.expander(f"{label} · {len(chunks)} chunks"):
            for i, ch in enumerate(chunks, 1):
                st.markdown(f"**{i}. {ch.get('source') or 'eBook'} — page {ch.get('page')}** · "
                            f"relevance `{ch['relevance_score']:.2f}`")
                st.text(ch["content"])


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif "error" in msg:
            st.error(msg["error"])
        else:
            render_assistant(msg["payload"])

if prompt := st.chat_input("Ask about the Agentic AI eBook..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching the eBook..."):
            result = call_api(prompt)
        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append({"role": "assistant", "error": result["error"]})
        else:
            render_assistant(result)
            st.session_state.messages.append({"role": "assistant", "payload": result})
