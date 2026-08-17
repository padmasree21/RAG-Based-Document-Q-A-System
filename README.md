RAG-Based Document Q&A System

An LLM-powered Question Answering system built on Retrieval-Augmented Generation (RAG), enabling accurate, source-grounded answers over private documents — with significantly reduced hallucinations.

Overview

Traditional LLMs often hallucinate when asked questions about content outside their training data. This project solves that problem by combining semantic retrieval with generative language models — retrieving the most relevant chunks of a private document corpus and grounding the LLM's response strictly in that retrieved context.

The result is a fast, accurate, and explainable Q&A system that can answer natural-language questions over any custom document set, with responses traceable back to their source.

Key Features
Semantic Retrieval Pipeline — Documents are chunked, embedded using dense vector embeddings, and indexed in ChromaDB for high-relevance similarity search.
Retrieval-Augmented Generation (RAG) — Combines retrieved context with an LLM to produce accurate, source-grounded answers instead of relying purely on model memory.
Reduced Hallucinations — Answers are constrained to retrieved document context, improving factual reliability over private/unseen data.
GPU-Accelerated Inference — Deployed with GPU support for low-latency response generation, even at scale.
Containerized Deployment — Fully Dockerized for reproducible, portable, and scalable deployment across environments.
Architecture
                 Document Corpus 
                └──────────┬───────────┘
                           │  Chunking
                           ▼
               
                  Embedding Model    
                 (Dense Vector Embed)
                └──────────┬───────────┘
                           │
                           ▼
                
                      ChromaDB      
                    (Vector Store)   
                └──────────┬───────────┘
                           │  Top-k Retrieval
                           ▼
      User Query ───▶ Retriever ───▶ Context Chunks
                           │
                           ▼
                
                          LLM           
                  (Answer Generation)  
                └──────────┬───────────┘
                           │
                           ▼
                Source-Grounded Answer
Tech Stack
Component	Technology
Retrieval	Dense embeddings + semantic search
Vector Database	ChromaDB
Generation	LLM (RAG pipeline)
Deployment	Docker (GPU-enabled)
Language	Python
Installation
Prerequisites
Docker & Docker Compose
NVIDIA GPU + CUDA drivers (for GPU-accelerated inference)
Python 3.10+
Setup
bash
# Clone the repository
git clone https://github.com/padmasree21/rag-document-qa.git
cd rag-document-qa

# Build the Docker image
docker build -t rag-doc-qa .

# Run the container with GPU support
docker run --gpus all -p 8000:8000 rag-doc-qa
Usage
Ingest Documents Place your source documents in the data/ directory. The pipeline will chunk and embed them automatically.
bash
   python ingest.py --path ./data
Installation
Prerequisites
Docker & Docker Compose
NVIDIA GPU + CUDA drivers (for GPU-accelerated inference)
Python 3.10+
Setup
bash
# Clone the repository
git clone https://github.com/padmasree21/rag-document-qa.git
cd rag-document-qa

# Build the Docker image
docker build -t rag-doc-qa .

# Run the container with GPU support
docker run --gpus all -p 8000:8000 rag-doc-qa
Usage

1. Ingest Documents

Place your source documents in the data/ directory. The pipeline will chunk and embed them automatically.

bash
python ingest.py --path ./data

2. Start the Q&A Service

bash
docker-compose up

3. Ask a Question

bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"What are the key findings in the Q3 report?\"}"

Example response:

json
{
  "answer": "...",
  "sources": ["report_q3.pdf - page 4", "report_q3.pdf - page 7"]
}


Pipeline Details
Chunking — Documents are split into overlapping semantic chunks to preserve context across boundaries.
Embedding — Each chunk is converted into a dense vector representation using a pretrained embedding model.
Indexing — Vectors are stored in ChromaDB for efficient approximate nearest-neighbor search.
Retrieval — At query time, the top-k most relevant chunks are retrieved based on semantic similarity.
Generation — Retrieved chunks are passed as context to the LLM, which generates a grounded, cited answer.
Results & Impact
Delivered accurate, source-grounded responses over private, previously unseen documents.
Reduced hallucination rate compared to a non-RAG baseline by grounding generation in retrieved context.
Achieved low-latency query response through GPU-enabled inference and containerized deployment.
Roadmap
Add support for multi-modal documents (tables, images)
Implement re-ranking for improved retrieval precision
Add streaming responses
Support for multiple LLM backends (local + API-based)
Web-based UI for interactive querying
Author

Padmasree Banoth Email: banothpadmasree@gmail.com | GitHub: padmasree21
