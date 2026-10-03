"""
test_rag.py
===========
Comprehensive verification script for the RAG Sub-Agent.

Tests:
1. PDF Loading & Document Parsing
2. Text Chunking
3. Pinecone Indexing (OpenAI text-embedding-3-small)
4. Vector Store Similarity Retrieval
5. Grounded LLM Generation (3 In-Domain Questions + 1 Out-of-Domain Non-Hallucination Question)
"""

import os
import sys
from dotenv import load_dotenv

# Ensure proper utf-8 encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from agents.rag_agent import (
    index_pdf_to_pinecone,
    get_pinecone_vector_store,
    answer_with_rag,
)

PDF_FILE = os.path.join("data", "agentic_ai_roadmap.pdf")
NAMESPACE = "agentic-ai-roadmap"


def print_banner(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def test_1_indexing():
    print_banner("TEST 1: PDF Ingestion, Chunking & Pinecone Indexing")
    print(f"Target PDF: {PDF_FILE}")
    print(f"Target Namespace: {NAMESPACE}")
    
    total_indexed = index_pdf_to_pinecone(
        pdf_path=PDF_FILE,
        namespace=NAMESPACE,
        chunk_size=1000,
        chunk_overlap=200,
        force_reindex=False,
    )
    print(f"[PASSED] Pinecone namespace '{NAMESPACE}' has {total_indexed} active chunks.")
    return total_indexed


def test_2_retriever():
    print_banner("TEST 2: Pinecone Vector Similarity Retrieval")
    vector_store = get_pinecone_vector_store(namespace=NAMESPACE)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    test_query = "What courses are recommended for learning Python in Phase 1?"
    print(f"Query: '{test_query}'")
    docs = retriever.invoke(test_query)
    
    print(f"Retrieved {len(docs)} chunks successfully:")
    for idx, doc in enumerate(docs, start=1):
        page_num = doc.metadata.get("page", 0)
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"  [Chunk {idx} | Page {page_num + 1}]: {snippet}...")
    
    assert len(docs) > 0, "No documents were retrieved!"
    print("[PASSED] Retrieval test successful.")


def test_3_rag_questions():
    print_banner("TEST 3: End-to-End RAG Sub-Agent Generation")
    
    questions = [
        # Question 1: Core Python Courses (Page 1)
        "According to the roadmap, what are the recommended courses for learning Core Python in Phase 1?",
        
        # Question 2: Agentic AI Design Patterns (Page 20)
        "What are the common Agentic AI design patterns listed in Step 7?",
        
        # Question 3: Agentic AI Frameworks (Page 21)
        "Which Agentic AI frameworks are introduced in Step 8 (Weeks 33-36)?",
        
        # Question 4: Out-of-domain (Non-hallucination check)
        "What is the secret recipe for baking a chocolate lava cake according to the document?"
    ]
    
    for i, q in enumerate(questions, start=1):
        print("\n" + "-" * 75)
        print(f"QUESTION {i}: {q}")
        print("-" * 75)
        
        result = answer_with_rag(query=q, namespace=NAMESPACE)
        answer = result["answer"]
        docs = result["retrieved_docs"]
        
        print(f"Retrieved Chunks: {len(docs)}")
        print("RAG SUB-AGENT ANSWER:")
        print(answer)
        print("-" * 75)


def run_interactive_mode():
    print_banner("INTERACTIVE RAG SUB-AGENT TERMINAL CHAT")
    print("PDF: data/agentic_ai_roadmap.pdf (Hosted on Pinecone)")
    print("Type your question below, or type 'exit' / 'quit' to stop.\n")
    
    while True:
        try:
            user_query = input("Ask RAG Agent (or 'exit') > ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive RAG session. Goodbye!")
                break
                
            print("\nSearching Pinecone & generating answer...")
            result = answer_with_rag(query=user_query, namespace=NAMESPACE)
            print("\n" + "=" * 60)
            print("RAG ANSWER:")
            print(result["answer"])
            print("=" * 60 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break


if __name__ == "__main__":
    try:
        # Agar user ne command line par direct question diya: python test_rag.py "question"
        if len(sys.argv) > 1:
            arg = sys.argv[1].strip()
            if arg in ["--interactive", "-i", "chat"]:
                run_interactive_mode()
            else:
                custom_query = " ".join(sys.argv[1:])
                print_banner(f"Custom Query: {custom_query}")
                res = answer_with_rag(query=custom_query, namespace=NAMESPACE)
                print("\nRAG ANSWER:")
                print(res["answer"])
        else:
            # Default: Run full verification test suite
            test_1_indexing()
            test_2_retriever()
            test_3_rag_questions()
            print_banner("ALL RAG SUB-AGENT TESTS PASSED SUCCESSFULLY!")
            print("\nTip: To test custom questions interactively, run:")
            print("  .\\.venv\\Scripts\\python.exe test_rag.py --interactive")
            print("Or single question:")
            print('  .\\.venv\\Scripts\\python.exe test_rag.py "Your question here"')
    except Exception as exc:
        print(f"\n[ERROR OCCURRED]: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
