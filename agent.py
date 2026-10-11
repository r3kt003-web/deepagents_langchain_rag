"""Agent initialization and subagent configuration module."""

import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.chat_models import init_chat_model
from prompts import (
    CHUNK_ANALYST_INSTRUCTIONS,
    get_system_prompt,
)
from tools import (
    backend as default_backend,
    create_search_documentation_tool,
    create_search_user_documents_tool,
    search_documentation as default_search_tool,
)

# Ensure API keys are loaded
load_dotenv()
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

MAX_CONCURRENT_ANALYSTS = 1

chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "Analyze one retrieved chunk file from user documents or documentation. "
        "Pass the user question and a single file path under /retrieved/."
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
}


def get_agent(
    model_name: str = "gemini-flash-lite-latest",
    user_vector_store=None,
    docs_vector_store=None,
    agent_backend: StateBackend | None = None,
):
    """Create and compile the deep agent with dynamic retrieval tools and subagents.

    Args:
        model_name: Name of the Gemini chat model.
        user_vector_store: Optional session-isolated vector store of user uploaded docs.
        docs_vector_store: Optional vector store of official documentation.
        agent_backend: Optional StateBackend instance for virtual filesystem isolation.

    Returns:
        Compiled DeepAgent graph ready for invocation.
    """
    model = init_chat_model(
        model=model_name,
        model_provider="google_genai",
        max_retries=6,
    )

    active_backend = agent_backend if agent_backend is not None else StateBackend()
    tools = []

    # If user has uploaded documents, register the user search tool
    if user_vector_store is not None:
        user_tool = create_search_user_documents_tool(
            user_vector_store=user_vector_store,
            tool_backend=active_backend,
        )
        tools.append(user_tool)

    # Register official documentation search tool
    docs_tool = create_search_documentation_tool(
        docs_vector_store=docs_vector_store,
        tool_backend=active_backend,
    )
    tools.append(docs_tool)

    # Build prompt tailored to whether user documents are loaded
    system_prompt = get_system_prompt(
        has_user_docs=(user_vector_store is not None),
        max_concurrent_analysts=MAX_CONCURRENT_ANALYSTS,
    )

    return create_deep_agent(
        model=model,
        tools=tools,
        backend=active_backend,
        system_prompt=system_prompt,
        subagents=[chunk_analyst_subagent],
    )


# Default agent instance
agent = get_agent()
