"""Agent tools and StateBackend for document and knowledge base retrieval."""

import time
import uuid
from typing import Any
from deepagents.backends import StateBackend
from langchain_core.tools import tool
from knowledge_base import get_vector_store

# Default module-level backend for CLI backwards-compatibility
backend = StateBackend()


def create_search_documentation_tool(
    docs_vector_store=None,
    tool_backend: StateBackend | None = None,
):
    """Factory creating a search tool for official LangChain and DeepAgents documentation."""
    active_backend = tool_backend or backend

    @tool(parse_docstring=True)
    def search_documentation(query: str) -> str:
        """Search official LangChain and DeepAgents technical documentation and save matching chunks to the agent filesystem.

        Args:
            query: Natural language technical search query.

        Returns:
            File paths where retrieved chunks were saved under /retrieved/, along with content excerpts.
        """
        store = docs_vector_store or get_vector_store()
        retrieved_docs = []

        for attempt in range(4):
            try:
                retrieved_docs = store.similarity_search(query, k=3)
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
            path = f"/retrieved/langchain_docs/{batch_id}/chunk_{index}.md"
            src = doc.metadata.get("source", "unknown")
            content = f"# Source: {src}\n\n{doc.page_content}"
            uploads.append((path, content.encode("utf-8")))
            saved_paths.append(path)
            chunk_previews.append(
                f"### [Chunk {index}] File: `{path}`\nSource: {src}\n\n{doc.page_content}"
            )

        active_backend.upload_files(uploads)
        return (
            f"Saved {len(saved_paths)} official documentation chunks under /retrieved/langchain_docs/{batch_id}/:\n"
            + "\n".join(saved_paths)
            + "\n\n--- Retrieved Documentation Chunks ---\n"
            + "\n\n".join(chunk_previews)
        )

    return search_documentation


def create_search_user_documents_tool(
    user_vector_store,
    tool_backend: StateBackend | None = None,
):
    """Factory creating a secure search tool for user-uploaded documents."""
    active_backend = tool_backend or backend

    @tool(parse_docstring=True)
    def search_user_documents(query: str) -> str:
        """Search the user's uploaded documents and save matching excerpts to the agent filesystem.

        Args:
            query: Natural language search query matching the user's uploaded document content.

        Returns:
            File paths where retrieved excerpts were saved under /retrieved/user_docs/, along with content excerpts.
        """
        retrieved_docs = []
        for attempt in range(4):
            try:
                retrieved_docs = user_vector_store.similarity_search(query, k=4)
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
            path = f"/retrieved/user_docs/{batch_id}/chunk_{index}.md"
            src = doc.metadata.get("source", "uploaded_file")
            page_info = f", Page: {doc.metadata['page']}" if "page" in doc.metadata else ""
            doc_type = doc.metadata.get("type", "document")

            # Format with strict untrusted data boundaries for prompt injection defense
            content = (
                f"# User Upload Source: {src} ({doc_type}{page_info})\n\n"
                f"<<<UNTRUSTED_USER_DOCUMENT_CHUNK Source: {src}{page_info}>>>\n"
                f"{doc.page_content}\n"
                f"<<<END_UNTRUSTED_USER_DOCUMENT_CHUNK>>>"
            )

            uploads.append((path, content.encode("utf-8")))
            saved_paths.append(path)
            chunk_previews.append(
                f"### [User Document Chunk {index}] File: `{path}`\n"
                f"Document: **{src}**{page_info}\n\n"
                f"<<<UNTRUSTED_USER_DOCUMENT_CHUNK Source: {src}{page_info}>>>\n"
                f"{doc.page_content}\n"
                f"<<<END_UNTRUSTED_USER_DOCUMENT_CHUNK>>>"
            )

        active_backend.upload_files(uploads)
        return (
            f"Saved {len(saved_paths)} user document excerpts under /retrieved/user_docs/{batch_id}/:\n"
            + "\n".join(saved_paths)
            + "\n\n--- Retrieved User Document Excerpts ---\n"
            + "\n\n".join(chunk_previews)
        )

    return search_user_documents


# Default search_documentation tool for backwards compatibility
search_documentation = create_search_documentation_tool(
    docs_vector_store=None, tool_backend=backend
)
