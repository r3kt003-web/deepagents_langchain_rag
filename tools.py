"""Agent tools and StateBackend for LangChain documentation retrieval."""

import time
import uuid
from deepagents.backends import StateBackend
from langchain_core.tools import tool
from knowledge_base import get_vector_store

backend = StateBackend()


@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search LangChain documentation and save matching chunks to the agent filesystem.

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/, along with content excerpts.
    """
    vector_store = get_vector_store()
    retrieved_docs = []

    # Retry loop with backoff in case of rate limits
    for attempt in range(4):
        try:
            retrieved_docs = vector_store.similarity_search(query, k=3)
            break
        except Exception as e:
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                time.sleep(10 * (attempt + 1))
            elif attempt == 3:
                raise e
            else:
                time.sleep(3)

    batch_id = uuid.uuid4().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []
    chunk_previews: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        src = doc.metadata.get("source", "unknown")
        content = (
            f"# Source: {src}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))
        saved_paths.append(path)
        chunk_previews.append(f"### [Chunk {index}] File: `{path}`\nSource: {src}\n\n{doc.page_content}")

    backend.upload_files(uploads)
    return (
        f"Saved {len(saved_paths)} documentation chunks under /retrieved/{batch_id}/:\n"
        + "\n".join(saved_paths)
        + "\n\n--- Retrieved Content ---\n"
        + "\n\n".join(chunk_previews)
    )
