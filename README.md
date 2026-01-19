# RAG Document Q&A with Reranking

A Retrieval-Augmented Generation application built for the Predusk_Tech  technical challenge. This app allows users to upload documents, indexes them into a vector database, and answers queries with strict citations using a two-stage retrieval process (Retrieval + Reranking).

## Architecture

1.  **Ingestion:** PDF/TXT -> LangChain Chunking -> OpenAI Embeddings -> Pinecone Serverless.
2.  **Retrieval:** Query -> OpenAI Embeddings -> Pinecone (Top-K=10).
3.  **Reranking:** Top-10 Docs -> Cohere Rerank v3 -> Top-3 Most Relevant.
4.  **Generation:** GPT-4o-mini -> Answer with Inline Citations.

## Tech Stack
* **Frontend:** Streamlit
* **Vector DB:** Pinecone (Serverless)
* **Reranker:** Cohere (`rerank-english-v3.0`)
* **LLM:** OpenAI (`gpt-4o-mini`)

## Setup & Run

1.  **Clone Repo:**
    ```bash
    git clone https://github.com/duck-relay/Test_Predusk_Tech
    cd Test_Predusk_Tech
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Rename `.env.example` to `.env` and add your API keys:
    * `OPENAI_API_KEY`
    * `PINECONE_API_KEY`
    * `COHERE_API_KEY`

4.  **Run App:**
    ```bash
    streamlit run app.py
    ```

### Minimal Evaluation (Gold Set)
*Test Document: "Attention Is All You Need" (Vaswani et al., 2017)*

| Query | Expected Answer (Gold Truth) | Actual Retrieval & Answer | Result |
| :--- | :--- | :--- | :--- |
| **"What is the Transformer?"** | A model architecture eschewing recurrence and relying entirely on an attention mechanism. | The model correctly identified the Transformer as a new network architecture based solely on attention mechanisms, citing [Source 1]. | ✅ Pass |
| **"Why is self-attention preferred?"** | It has a constant number of sequentially executed operations and connects all positions. | Correctly retrieved chunks discussing computational complexity and parallelization. Answered that it allows for more parallelization. | ✅ Pass |
| **"How many layers are in the encoder?"** | The encoder is composed of a stack of N=6 identical layers. | Answer: "The encoder consists of a stack of N = 6 identical layers." Correctly cited [Source 2]. | ✅ Pass |
| **"Which optimizer was used?"** | The Adam optimizer with beta1=0.9, beta2=0.98. | Retrieved the "Optimizer" section. Answer specifically mentioned the Adam optimizer and correct beta values. | ✅ Pass |
| **"What is the recipe for pizza?"** | (Irrelevant / No Answer) | The model correctly refused to answer, stating the document does not contain this information. | ✅ Pass |

**Observation:** The system demonstrated high precision on factual queries. The reranker successfully prioritized the "Optimizer" section (Chunk 12) over less relevant mentions of parameters in the appendix.

## Remarks & Trade-offs
* **Storage:** Used local filesystem for temporary upload processing. In production, I would use S3.
*### 2. Embeddings & Chunking
* **Model:** OpenAI `text-embedding-3-small` (1536 dimensions).
* **Chunking Strategy:**
    * **Size:** 4000 characters (approx. 1000 tokens).
    * **Overlap:** 400 characters (approx. 100 tokens).
    * **Reasoning:** Chosen to fit the "800–1,200 token" requirement while maintaining enough context window for the LLM to understand sentence boundaries.
* **History:** Chat history is session-based (RAM). A permanent database (Postgres) is needed for persistent history.