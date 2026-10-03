"""
agents/rag_agent.py
===================
RAG (Retrieval-Augmented Generation) Sub-Agent using Hosted Pinecone Vector Database.

LangChain RAG Pipeline Flow:
1. PDF Loading        -> PyPDFLoader loads the PDF document.
2. Text Splitting     -> RecursiveCharacterTextSplitter splits document into semantically coherent chunks.
3. Embedding          -> OpenAIEmbeddings (text-embedding-3-small) generates vector representations.
4. Hosted Vector Store-> PineconeVectorStore stores chunks in the Pinecone cloud database.
5. Retriever          -> Vector store similarity retriever fetches the most relevant chunks.
6. Sub-Agent Graph    -> LangGraph StateGraph orchestrates retrieval, context formatting, and LLM answer generation.
7. Grounded Answer    -> The LLM (gpt-4o-mini) answers strictly based on retrieved PDF context.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

# Windows console UTF-8 support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain Core & OpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Document Loaders & Splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hosted Vector Store (Pinecone)
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# LangGraph
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()


# ==============================================================================
# 1. State Definition for RAG Sub-Agent
# ==============================================================================
class RAGSubAgentState(TypedDict):
    """
    State object passed between nodes in the RAG Sub-Agent graph.
    - query: User's original question
    - retrieved_docs: List of Document chunks fetched from Pinecone
    - context_text: Concatenated text of retrieved chunks
    - answer: Final grounded answer produced by the LLM
    """
    query: str
    retrieved_docs: List[Document]
    context_text: str
    answer: str


# ==============================================================================
# 2. Vector Store & Indexing Functions
# ==============================================================================
def get_embeddings() -> OpenAIEmbeddings:
    """Returns OpenAI embeddings model (text-embedding-3-small, dimension 1536)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured in .env file.")
    return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)


def get_llm() -> ChatOpenAI:
    """Returns the chat model for generating grounded answers."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured in .env file.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)


def get_pinecone_vector_store(namespace: str = "agentic-ai-roadmap") -> PineconeVectorStore:
    """Connects to the hosted Pinecone vector store."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "langgraph-rag-index")
    
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not configured in .env file.")
        
    embeddings = get_embeddings()
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key,
        namespace=namespace,
    )


def index_pdf_to_pinecone(
    pdf_path: str,
    namespace: str = "agentic-ai-roadmap",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    force_reindex: bool = False,
) -> int:
    """
    Loads a PDF, normalizes its text (cleaning multi-newlines and broken spacing),
    chunks it into semantically coherent segments, embeds them, and uploads them to Pinecone.
    Skips uploading if vectors already exist in the namespace unless force_reindex=True.
    Returns the number of chunks indexed.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "langgraph-rag-index")
    
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    
    # Check if this namespace already has vectors
    namespace_stats = stats.get("namespaces", {}).get(namespace, {})
    existing_vectors = namespace_stats.get("vector_count", 0)
    
    if existing_vectors > 0 and not force_reindex:
        print(f"[RAG] Pinecone namespace '{namespace}' already contains {existing_vectors} vectors. Skipping re-upload.")
        return existing_vectors

    if force_reindex and existing_vectors > 0:
        print(f"[RAG] Deleting existing {existing_vectors} vectors from namespace '{namespace}' for clean re-indexing...")
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception as e:
            print(f"[RAG] Warning on namespace delete: {e}")

    print(f"[RAG] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"[RAG] Loaded {len(pages)} pages from PDF.")

    # Normalize whitespace on every page to prevent fragmented word-by-word newlines
    cleaned_pages = []
    pdf_basename = os.path.basename(pdf_path)
    for p in pages:
        normalized_content = " ".join(p.page_content.split())
        if normalized_content:
            p.page_content = normalized_content
            p.metadata["source"] = pdf_basename
            cleaned_pages.append(p)

    print(f"[RAG] Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(cleaned_pages)
    print(f"[RAG] Created {len(chunks)} cleaned text chunks.")

    print(f"[RAG] Generating OpenAI embeddings and indexing to Pinecone index '{index_name}' (namespace='{namespace}')...")
    embeddings = get_embeddings()
    
    # Upload to Pinecone in batches to avoid network limits
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name,
            pinecone_api_key=pinecone_api_key,
            namespace=namespace,
        )
        print(f"[RAG] Uploaded batch {i + 1} to {min(i + batch_size, len(chunks))} of {len(chunks)}")

    print(f"[RAG] Successfully indexed {len(chunks)} chunks into hosted Pinecone!")
    return len(chunks)


def index_all_pdfs_in_data_dir(data_dir: str = "data", namespace: str = "agentic-ai-roadmap") -> Dict[str, int]:
    """
    Discovers all PDF documents in `data_dir` and ensures they are indexed in Pinecone.
    """
    if not os.path.exists(data_dir):
        return {}
        
    results = {}
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            count = index_pdf_to_pinecone(pdf_path, namespace=namespace, force_reindex=False)
            results[filename] = count
    return results


# ==============================================================================
# 3. LangGraph RAG Sub-Agent Nodes
# ==============================================================================
def create_retrieve_node(vector_store: PineconeVectorStore, top_k: int = 6):
    """
    Node 1: Retrieves relevant chunks from Pinecone based on user's query.
    If query is an overview or broad inquiry, enriches retrieval with roadmap index chunks.
    Updates state: retrieved_docs, context_text
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    def retrieve_node(state: RAGSubAgentState) -> Dict[str, Any]:
        query = state["query"]
        docs = list(retriever.invoke(query))
        
        # Overview query enrichment: If user asks what is in the PDF, saved data, or overall roadmap
        q_lower = query.lower()
        overview_terms = ["saved pdf", "about your pdf", "in agent ai pdf", "in the pdf", "all information", "overview", "what is this pdf", "summarize", "about your saved"]
        if any(term in q_lower for term in overview_terms):
            try:
                overview_docs = vector_store.similarity_search("The Ultimate Roadmap to Learning Agentic AI Comprehensive Step-by-Step Overview Phases", k=3)
                existing_texts = {d.page_content for d in docs}
                for od in overview_docs:
                    if od.page_content not in existing_texts:
                        docs.append(od)
                        existing_texts.add(od.page_content)
            except Exception:
                pass
        
        # Build formatted context text from retrieved document chunks
        context_parts = []
        for idx, doc in enumerate(docs, start=1):
            source_page = doc.metadata.get("page", "Unknown")
            source_file = doc.metadata.get("source", "PDF Document")
            page_num = source_page + 1 if isinstance(source_page, int) else source_page
            context_parts.append(
                f"[Document Chunk {idx} | Source: {source_file} | Page {page_num}]:\n{doc.page_content.strip()}"
            )
        context_text = "\n\n".join(context_parts)
        
        return {
            "retrieved_docs": docs,
            "context_text": context_text,
        }

    return retrieve_node


def create_generate_node(llm: ChatOpenAI):
    """
    Node 2: Generates a grounded, comprehensive response using the retrieved context.
    Synthesizes roadmap phases, topics, tools, and components naturally.
    Updates state: answer
    """
    system_prompt = (
        "You are an expert RAG (Retrieval-Augmented Generation) knowledge assistant.\n"
        "Your task is to answer the user's questions accurately, comprehensively, and helpfully based on the provided Context from the indexed document(s) (such as the Agentic AI Roadmap PDF).\n\n"
        "GUIDELINES:\n"
        "1. GROUNDED KNOWLEDGE: Answer using the facts, topics, phases, steps, tools, architectures, and concepts described in the Context.\n"
        "2. INTELLIGENT SYNTHESIS: When the user asks about components, architectures, skills, or roadmap steps (e.g. core components of an AI agent, foundational knowledge, etc.), synthesize the relevant points listed in the context (e.g. autonomy, perception, decision-making, action, memory, reflex agents, goal-based agents, etc.) and explain how the document presents them.\n"
        "3. OVERVIEW / SUMMARY QUESTIONS: If the user asks about the saved PDF data, what the document is about, or asks for all information/overview, provide a clear, well-structured summary of the roadmap phases, major topics, and target goals covered in the context.\n"
        "4. CITATIONS: Clearly mention the relevant Page numbers, Phases (Phase 1, 2, 3, 4), or Steps (e.g. Step 6 Agentic AI Fundamentals) from the document chunks.\n"
        "5. UNRELATED TOPICS: Only state that the document does not contain information if the query is about something completely unrelated to the document's topics.\n"
        "6. STRICT LANGUAGE RULE: ALWAYS detect the user's language and respond in the EXACT same language:\n"
        "   * If the user asks in English -> Respond STRICTLY in English. Do NOT use Roman Urdu or Hindi.\n"
        "   * If the user asks in Roman Urdu / Roman English -> Respond in Roman Urdu.\n"
        "   * If the user asks in Urdu script -> Respond in Urdu script."
    )

    user_template = (
        "Context from Document:\n"
        "---------------------\n"
        "{context}\n"
        "---------------------\n\n"
        "User Question: {question}\n\n"
        "Grounded Answer:"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_template),
    ])

    chain = prompt | llm

    def generate_node(state: RAGSubAgentState) -> Dict[str, Any]:
        query = state["query"]
        context_text = state.get("context_text", "")

        # If no context was retrieved at all
        if not context_text.strip():
            return {
                "answer": "The provided document does not contain any relevant information to answer this question."
            }

        response = chain.invoke({
            "context": context_text,
            "question": query,
        })

        return {"answer": response.content}

    return generate_node


# ==============================================================================
# 4. Assembled RAG Sub-Agent Graph
# ==============================================================================
def build_rag_sub_graph(namespace: str = "agentic-ai-roadmap", top_k: int = 5):
    """
    Constructs and compiles the RAG Sub-Agent LangGraph.
    
    Graph Topology:
    [START] -> [retrieve] -> [generate] -> [END]
    """
    vector_store = get_pinecone_vector_store(namespace=namespace)
    llm = get_llm()

    workflow = StateGraph(RAGSubAgentState)

    # Add Nodes
    workflow.add_node("retrieve", create_retrieve_node(vector_store, top_k=top_k))
    workflow.add_node("generate", create_generate_node(llm))

    # Add Edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile the graph
    app = workflow.compile()
    return app


# ==============================================================================
# 5. Public Sub-Agent Interface Function
# ==============================================================================
def answer_with_rag(
    query: str,
    namespace: str = "agentic-ai-roadmap",
    history: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    High-level entry point to invoke the RAG sub-agent with a query.
    Returns the updated state containing query, retrieved_docs, context_text, and answer.
    """
    rag_app = build_rag_sub_graph(namespace=namespace)
    initial_state: RAGSubAgentState = {
        "query": query,
        "retrieved_docs": [],
        "context_text": "",
        "answer": "",
    }
    final_state = rag_app.invoke(initial_state)
    return final_state
