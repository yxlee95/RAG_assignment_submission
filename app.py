from __future__ import annotations

import streamlit as st

from src.config import AppConfig
from src.embeddings import EmbeddingModel
from src.llm_client import GeminiClient
from src.rag_pipeline import RAGChatbot
from src.vector_store import NumpyVectorStore


@st.cache_resource(show_spinner=False)
def get_chatbot() -> RAGChatbot:
    config = AppConfig()

    store = NumpyVectorStore()
    store.load(config.index_vectors_path, config.index_records_path)

    embedding_model = EmbeddingModel(config.embedding_model)
    llm_client = GeminiClient(
        api_key=config.gemini_api_key,
        model=config.gemini_model,
        verify_ssl=config.gemini_verify_ssl,
        ca_bundle=config.gemini_ca_bundle,
    )

    return RAGChatbot(
        embedding_model=embedding_model,
        vector_store=store,
        llm_client=llm_client,
        top_k=config.top_k,
    )


def main() -> None:
    st.set_page_config(page_title="RAG Chatbot Assessment", page_icon="💬", layout="centered")
    st.title("💬 Recruitment Assessment: RAG Chatbot")
    st.caption("Knowledge base: Lesson_Learned_Sample data file")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about the lesson learned data")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        chatbot = get_chatbot()
    except Exception as exc:
        error_msg = (
            "Initialization failed. Ensure index files exist and GEMINI_API_KEY is set.\n\n"
            f"Error: {exc}"
        )
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        return

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving context and generating answer..."):
                result = chatbot.ask(user_query)
        except Exception as exc:
            error_msg = f"Request failed. {exc}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            return

        st.markdown(result.answer)

        if result.retrieved_chunks:
            with st.expander("Retrieved Context"):
                for i, chunk in enumerate(result.retrieved_chunks, start=1):
                    source = chunk.metadata.get("source_file", "unknown")
                    row = chunk.metadata.get("row_index", "N/A")
                    st.markdown(f"**Chunk {i}** | source: {source} | row: {row} | score: {chunk.score:.4f}")
                    st.write(chunk.text)

    st.session_state.messages.append({"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    main()
