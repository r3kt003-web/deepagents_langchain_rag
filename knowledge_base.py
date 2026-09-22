"""Knowledge base module for fetching, splitting, embedding, and caching LangChain docs."""

import os
import time
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure environment variables are loaded
load_dotenv()
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

DOCS_BASE = "https://docs.langchain.com"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "vectorstore_cache.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "docs_data")

# Core documentation pages focused on deepagents, subagents, streaming, and langchain
DOC_PATHS = [
    "oss/python/deepagents/rag",
    "oss/python/deepagents/subagents",
    "oss/python/deepagents/streaming",
    "oss/python/deepagents/frontend/subagent-streaming",
    "oss/python/deepagents/backends",
    "oss/python/langchain/agents",
    "oss/python/langchain/tools",
]


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initialize the Google Generative AI Embeddings model."""
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")


def load_langchain_docs(doc_paths: list[str] | None = None) -> list[Document]:
    """Fetch or load locally cached LangChain documentation pages as Documents."""
    paths = doc_paths or DOC_PATHS
    docs: list[Document] = []
    os.makedirs(DATA_DIR, exist_ok=True)

    for path in paths:
        fname = path.replace("/", "_") + ".md"
        local_path = os.path.join(DATA_DIR, fname)
        source = f"{DOCS_BASE}/{path}"

        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": source}))
            continue

        url = f"{DOCS_BASE}/{path}.md"
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                docs.append(Document(page_content=response.text, metadata={"source": source}))
            else:
                print(f"Skipping {url} (status: {response.status_code})")
        except requests.RequestException as e:
            print(f"Warning: Failed to fetch {url}: {e}")
            continue

    return docs


def build_vector_store(
    doc_paths: list[str] | None = None,
    chunk_size: int = 3500,
    chunk_overlap: int = 250,
) -> InMemoryVectorStore:
    """Fetch documentation, split into chunks, embed with rate-limit retries, and dump to disk."""
    docs = load_langchain_docs(doc_paths)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(docs)
    print(f"Loaded {len(docs)} documents and split into {len(all_splits)} chunks.", flush=True)

    embeddings = get_embeddings()
    vector_store = InMemoryVectorStore(embeddings)

    batch_size = 40
    total_splits = len(all_splits)

    for i in range(0, total_splits, batch_size):
        batch = all_splits[i : i + batch_size]
        max_retries = 5
        success = False

        for attempt in range(max_retries):
            try:
                vector_store.add_documents(documents=batch)
                print(f"Indexed chunks {i + 1} to {min(i + batch_size, total_splits)} of {total_splits}...", flush=True)
                # Incremental persistence so progress is never lost
                try:
                    vector_store.dump(CACHE_PATH)
                except Exception:
                    pass
                success = True
                time.sleep(2.0)
                break
            except Exception as e:
                err_msg = str(e)
                if any(x in err_msg for x in ["RESOURCE_EXHAUSTED", "429", "502", "503", "504", "Bad Gateway", "quota"]):
                    wait_time = 25 * (attempt + 1)
                    print(f"Temporary API throttling / server error ({e}). Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"Unexpected embedding error: {e}", flush=True)
                    time.sleep(5)

        if not success:
            print(f"Warning: Batch {i} to {i + batch_size} could not be indexed after retries. Continuing with remaining chunks...", flush=True)

    # Final dump to ensure all indexed items are persisted
    try:
        vector_store.dump(CACHE_PATH)
        print(f"Successfully persisted vector store cache to {CACHE_PATH}", flush=True)
    except Exception as e:
        print(f"Warning: Could not save vector store to cache: {e}", flush=True)

    return vector_store


_cached_store: InMemoryVectorStore | None = None


def get_vector_store(force_refresh: bool = False) -> InMemoryVectorStore:
    """Return the vector store, loading from disk cache if available or building if needed."""
    global _cached_store
    if _cached_store is not None and not force_refresh:
        return _cached_store

    embeddings = get_embeddings()

    if not force_refresh and os.path.exists(CACHE_PATH):
        try:
            print(f"Loading vector store from disk cache: {CACHE_PATH}", flush=True)
            _cached_store = InMemoryVectorStore.load(CACHE_PATH, embeddings)
            return _cached_store
        except Exception as e:
            print(f"Error loading cache ({e}). Rebuilding vector store...", flush=True)

    _cached_store = build_vector_store()
    return _cached_store
