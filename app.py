import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from datetime import datetime

# Load environment
load_dotenv()
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = openai_api_key

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate

# Streamlit setup
st.set_page_config(page_title="Chat CCU", page_icon="🎓")

DATA_DIR = "data"
PERSIST_DIR = "storage"

# LLM & embeddings
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_key=openai_api_key
)
Settings.llm = OpenAI(
    model="gpt-4o",
    temperature=0.3,
    api_key=openai_api_key,
    system_prompt="""
    You are a helpful university assistant. Use only the provided university PDF documents to answer questions.
    Always cite your sources in this format: (DocumentName, page X).
    If the answer is not in the context, say: "I don't know the answer."
    Be clear, concise, and NEVER guess.
    """
)

# Prompt template
custom_template = PromptTemplate(
    "Use only the context below to answer the question.\n"
    "If listing items, format them as bullet points using markdown (- Item 1, - Item 2, etc).\n"
    "Always cite the document name and page number like this: (DocumentName, page X).\n"
    "If the answer is not in the context, say 'I don't know the answer.' Do not guess.\n\n"
    "Question: {query_str}\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Answer:"
)

# Load or build index
@st.cache_resource
def load_index():
    if not os.path.exists(PERSIST_DIR):
        from llama_index.core.node_parser import SentenceSplitter
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        nodes = splitter.get_nodes_from_documents(documents)

        index = VectorStoreIndex(nodes, show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

index = load_index()

# Setup retrievers
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=25)
bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=10)

# Routing logic
def get_retriever_for_query(query: str):
    sparse_keywords = ["email", "form", "ferpa", "gpa", "deadline"]
    if any(kw in query.lower() for kw in sparse_keywords):
        return bm25_retriever
    return vector_retriever

# Postprocessor
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.15)

# Streamlit UI
# Apply custom Open Sans font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

    html, body, .stApp, .block-container, .main,
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stText, .stTextInput, .stTextArea, .stButton, .stSelectbox,
    .stCaption, .stRadio, .stSlider, .stDataFrame, .stNumberInput, .stExpander,
    label, div[data-testid="stMarkdownContainer"],
    div[data-testid="stExpander"] * {
        font-family: 'Open Sans', sans-serif !important;
    }
    /* Customize the expander background */
    div[data-testid="stExpander"] {
    background-color: #4D4D4F !important; /* Background color for the dropdown box */
    border: 1px solid #FED925; /* Border around the dropdown box */
    border-radius: 10px;
    }
    img {
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    border-radius: 8px;
    margin-bottom: 20px;
    }

    input, textarea {
    padding: 12px !important;
    border-radius: 8px !important;
    }


    </style>
""", unsafe_allow_html=True)



st.title("Welcome to Chat CCU")
st.markdown("")
st.image("logo.png", width=700)
with st.expander("How to use this tool"):
    st.markdown("""
    - Chat CCU can answer general questions from the Student Handbook, Course Catalog, Academic Policies, Financial Aid Handbook, and Graduation 2025 Info.
    - Be as specific as possible and avoid using abbreviations when asking questions.
    """)
st.markdown("**DISCLAIMER:** This tool does not replace academic advising. For official questions, please contact your advisor.")

query = st.text_input("Ask a question:", key="question_input", placeholder="Example: What are my graduation requirements?")
st.components.v1.html(
    f"""<script>
    const input = window.parent.document.querySelector('input[data-testid="stTextInput"][aria-label="Ask a question:"]');
    if (input) {{ input.focus(); }}
    </script>""",
    height=0,
)

if query:
    with st.spinner("Thinking..."):
        retriever = get_retriever_for_query(query)
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[postprocessor],
            text_qa_template=custom_template
        )
        response = query_engine.query(query)
    st.markdown("### 💬 Answer")
    st.markdown(response.response)

    # --- FEEDBACK SECTION ---
    feedback_text = st.text_area("Was this response helpful? Let us know:", key=query, height=60)

    if st.button("Submit Feedback", key=f"submit_{query}"):
        if feedback_text.strip():
            feedback_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response.response,
                "feedback": feedback_text.strip(),
            }

            # Append to CSV
            feedback_file = "feedback.csv"
            try:
                if os.path.exists(feedback_file):
                    df = pd.read_csv(feedback_file)
                    df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
                else:
                    df = pd.DataFrame([feedback_data])
                df.to_csv(feedback_file, index=False)
                st.success("✅ Feedback submitted. Thank you!")
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")
        else:
            st.warning("Please type something before submitting.")




