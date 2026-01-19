import streamlit as st
import os
import time
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
import cohere
from dotenv import load_dotenv

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="RAG Challenge (Track B)", layout="wide")

# Initialize Keys (Check env or Streamlit secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-app")

# Initialize Clients
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
if COHERE_API_KEY:
    co = cohere.Client(COHERE_API_KEY)

# --- 2. BACKEND FUNCTIONS ---

def ingest_file(uploaded_file):
    """Handles file loading, chunking, and upserting to Pinecone."""
    start_time = time.time()
    
    # Save temp file
    temp_filename = f"temp_{uploaded_file.name}"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Load
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_filename)
        else:
            loader = TextLoader(temp_filename)
        documents = loader.load()

        # Chunk (Requirement: 800-1200 tokens, 10-15% overlap)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=1000,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # Add Metadata for Citations
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["source"] = uploaded_file.name

        # Embed & Upsert
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Check/Create Index
        existing_indexes = [index.name for index in pc.list_indexes()]
        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536, # OpenAI small dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        index = pc.Index(INDEX_NAME)
        
        # Batch Upsert
        vectors = []
        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk.page_content)
            vectors.append({
                "id": f"{uploaded_file.name}_{i}",
                "values": vector,
                "metadata": {"text": chunk.page_content, "source": chunk.metadata["source"], "chunk_id": i}
            })
            
            # Upsert in batches of 100
            if len(vectors) >= 100:
                index.upsert(vectors)
                vectors = []
        
        if vectors:
            index.upsert(vectors)

        os.remove(temp_filename)
        return len(chunks), time.time() - start_time
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return 0, 0

def retrieve_and_rerank(query):
    """
    1. Retrieve top-k (10) from Pinecone.
    2. Rerank top-n (3) using Cohere.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vec = embeddings.embed_query(query)
    index = pc.Index(INDEX_NAME)

    # 1. Retrieval
    results = index.query(vector=query_vec, top_k=10, include_metadata=True)
    initial_docs = [match['metadata']['text'] for match in results['matches']]
    initial_metas = [match['metadata'] for match in results['matches']]

    if not initial_docs:
        return [], 0.0

    # 2. Reranking (Requirement for Track B)
    rerank_results = co.rerank(
        query=query,
        documents=initial_docs,
        top_n=3,
        model='rerank-english-v3.0'
    )

    # Reconstruct sorted results
    final_results = []
    for result in rerank_results.results:
        final_results.append({
            "text": initial_docs[result.index],
            "meta": initial_metas[result.index],
            "score": result.relevance_score
        })
    
    return final_results

def generate_answer(query, context_chunks):
    """Generates answer using GPT-4o-mini with citations."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Format context with citation IDs
    context_text = ""
    for i, chunk in enumerate(context_chunks):
        context_text += f"[Source {i+1}]: {chunk['text']}\n\n"

    system_prompt = f"""
    You are a helpful assistant. Use the provided context to answer the user's question.
    
    RULES:
    1. You MUST cite your sources using square brackets, e.g., [1] or [2].
    2. The context provided has Source IDs (e.g., [Source 1]). Map these to your citations.
    3. If the answer is not in the context, politely state that you cannot answer based on the provided documents.
    4. Keep answers concise.
    
    CONTEXT:
    {context_text}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    response = llm.invoke(messages)
    return response.content

# --- 3. FRONTEND UI ---

st.title("⚡ RAG Challenge: Doc Q&A with Reranking")
st.markdown("Retrieval (Pinecone) → Rerank (Cohere) → Answer (GPT-4o)")

# Sidebar for Setup
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    
    if uploaded_file and st.button("Ingest Document"):
        with st.spinner("Chunking & Embedding..."):
            count, duration = ingest_file(uploaded_file)
            if count > 0:
                st.success(f"Indexed {count} chunks in {duration:.2f}s!")
            else:
                st.error("Ingestion failed. Check logs.")

    st.divider()
    st.info("API Keys loaded from environment variables.")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process
    with st.chat_message("assistant"):
        start_t = time.time()
        
        # A. Retrieve & Rerank
        relevant_chunks = retrieve_and_rerank(prompt)
        
        if not relevant_chunks:
            response_text = "I couldn't find any relevant information in the document."
            st.markdown(response_text)
        else:
            # B. Generate
            response_text = generate_answer(prompt, relevant_chunks)
            st.markdown(response_text)
            
            # C. Show Citations & Metrics
            end_t = time.time()
            with st.expander(f"📚 Citations & Metrics ({end_t - start_t:.2f}s)"):
                st.caption(f"Cost Estimate: ~${(len(prompt + response_text)/1000) * 0.0005:.5f}")
                for i, chunk in enumerate(relevant_chunks):
                    st.markdown(f"**[{i+1}] Relevance: {chunk['score']:.4f}**")
                    st.text(chunk['text'][:200] + "...")

    st.session_state.messages.append({"role": "assistant", "content": response_text})
