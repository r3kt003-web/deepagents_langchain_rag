import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from prompts import (
    CHUNK_ANALYST_INSTRUCTIONS,
    RAG_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from tools import backend, search_documentation

# Ensure API keys are loaded
load_dotenv()
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

MAX_CONCURRENT_ANALYSTS = 1

INSTRUCTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts=MAX_CONCURRENT_ANALYSTS,
    )
)

chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "Analyze one retrieved documentation chunk file. "
        "Pass the user question and a single file path under /retrieved/."
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
}


def get_agent(model_name: str = "gemini-flash-lite-latest"):
    """Create and compile the deep agent with retrieval tools and subagents."""
    model = init_chat_model(
        model=model_name,
        model_provider="google_genai",
        max_retries=6,
    )

    return create_deep_agent(
        model=model,
        tools=[search_documentation],
        backend=backend,
        system_prompt=INSTRUCTIONS,
        subagents=[chunk_analyst_subagent],
    )


# Default agent instance
agent = get_agent()
