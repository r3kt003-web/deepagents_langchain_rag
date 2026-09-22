"""Main application entrypoint for the DeepAgents LangChain RAG system.

Supports both a Streamlit web interface and CLI execution.
To launch the web interface:
    streamlit run main.py
To run via CLI:
    python main.py --cli [query]
"""

import os
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent import get_agent
from knowledge_base import CACHE_PATH, DOC_PATHS, DOCS_BASE, get_vector_store

EXAMPLE_QUERY = "How do i stream intermediate tool results from a subagent"

AVAILABLE_MODELS = [
    ("gemini-flash-lite-latest", "⚡ Flash-Lite Latest (Recommended - High Quota)"),
    ("gemini-3.1-flash-lite", "💡 Gemini 3.1 Flash-Lite"),
    ("gemini-3.6-flash", "🚀 Gemini 3.6 Flash"),
    ("gemini-2.5-flash", "⚠️ Gemini 2.5 Flash (Strict Daily Limit)"),
]


def extract_message_text(content) -> str:
    """Extract readable text from message content, handling string or list-of-dicts."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content else ""


# ==============================================================================
# STREAMLIT UI IMPLEMENTATION
# ==============================================================================


def apply_custom_styles():
    """Inject modern styling with Google Fonts, glassmorphism, and vibrant accents."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        h1, h2, h3, h4, .main-title {
            font-family: 'Outfit', sans-serif !important;
            letter-spacing: -0.02em;
        }

        code, pre {
            font-family: 'Fira Code', monospace !important;
        }

        /* Hero Header */
        .hero-container {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 24px;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
            position: relative;
            overflow: hidden;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a5b4fc 0%, #38bdf8 50%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 6px 0;
        }

        .hero-subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
            margin: 0;
            font-weight: 400;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.25);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: #34d399;
            border-radius: 50%;
            box-shadow: 0 0 8px #34d399;
            display: inline-block;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0b1120;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        .sidebar-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 16px;
        }

        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 1.05rem;
            font-weight: 700;
            color: #f1f5f9;
        }

        .stButton button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_streamlit_page():
    """Configure Streamlit layout and session state."""
    st.set_page_config(
        page_title="DeepAgents | LangChain RAG",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemini-flash-lite-latest"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state or st.session_state.get("current_agent_model") != st.session_state.selected_model:
        st.session_state.agent = get_agent(model_name=st.session_state.selected_model)
        st.session_state.current_agent_model = st.session_state.selected_model
    if "selected_query" not in st.session_state:
        st.session_state.selected_query = None


def render_sidebar():
    """Render configuration and status in the sidebar."""
    with st.sidebar:
        st.markdown("### ⚡ Engine Overview")
        cache_exists = os.path.exists(CACHE_PATH)
        cache_status = "Cached on Disk (80 chunks)" if cache_exists else "Memory Only"
        cache_color = "#34d399" if cache_exists else "#fbbf24"

        # Model Selection Dropdown
        model_options = [m[0] for m in AVAILABLE_MODELS]
        model_labels = {m[0]: m[1] for m in AVAILABLE_MODELS}

        chosen_model = st.selectbox(
            "Select LLM Model",
            options=model_options,
            format_func=lambda x: model_labels.get(x, x),
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
            help="gemini-3.6-flash is recommended for reliable tool calling and synthesis.",
        )

        if chosen_model != st.session_state.selected_model:
            st.session_state.selected_model = chosen_model
            with st.spinner(f"Switching agent to {chosen_model}..."):
                st.session_state.agent = get_agent(model_name=chosen_model)
                st.session_state.current_agent_model = chosen_model
            st.rerun()

        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="metric-label">System Status</div>
                <div style="color: #34d399; font-weight: 600; font-size: 0.95rem;">
                    <span class="pulse-dot"></span> Ready & Operational
                </div>
                <div style="margin-top: 10px;" class="metric-label">Vector Cache</div>
                <div style="color: {cache_color}; font-weight: 600; font-size: 0.9rem;">{cache_status}</div>
                <div style="margin-top: 10px;" class="metric-label">Active Model</div>
                <div class="metric-value">{st.session_state.selected_model}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Re-index", help="Re-fetch docs and rebuild the vector cache"):
                with st.spinner("Rebuilding vector store cache..."):
                    get_vector_store(force_refresh=True)
                    st.success("Cache rebuilt!")
                    st.rerun()
        with col_b:
            if st.button("🧹 Clear Chat", help="Reset chat message history"):
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")
        st.markdown("### 💡 Quick Prompt Starters")
        st.caption("Click a question to test the RAG subagent pipeline:")

        sample_prompts = [
            ("Intermediate Subagent Streaming", EXAMPLE_QUERY),
            ("StateBackend Details", "How does StateBackend work in deepagents?"),
            ("Agent Creation Workflow", "How do I create a DeepAgent with custom tools?"),
            ("Recursive Text Splitter", "What chunk size and overlap does RecursiveCharacterTextSplitter recommend?"),
        ]

        for label, q in sample_prompts:
            if st.button(f"📌 {label}", key=f"btn_{label}", use_container_width=True):
                st.session_state.selected_query = q
                st.rerun()

        st.markdown("---")
        with st.expander("📚 Indexed Documentation Pages", expanded=False):
            st.caption(f"{len(DOC_PATHS)} official LangChain/DeepAgents docs indexed:")
            for p in DOC_PATHS:
                doc_name = p.split("/")[-1].replace("-", " ").title()
                st.markdown(f"- [{doc_name}]({DOCS_BASE}/{p})")


def render_chat_interface():
    """Render header and interactive chat history."""
    st.markdown(
        """
        <div class="hero-container">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h1 class="hero-title">DeepAgents LangChain RAG</h1>
                    <p class="hero-subtitle">Multi-agent documentation assistant powered by Gemini, LangChain vector search, and subagent delegation.</p>
                </div>
                <div>
                    <div class="status-pill">
                        <span class="pulse-dot"></span> Multi-Agent Active
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display chat history
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        intermediate = msg.get("intermediate", {})

        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "⚡"):
            st.markdown(content)

            # If agent response has intermediate steps or retrieved files, display expandable inspector
            if intermediate and role == "assistant":
                with st.expander("🔍 Inspection & Retrieved Chunks", expanded=False):
                    tabs = st.tabs(["📂 Saved Files (/retrieved/)", "🛠️ Tool & Subagent Activity"])
                    with tabs[0]:
                        files = intermediate.get("files", {})
                        if files:
                            for file_path, file_data in files.items():
                                st.markdown(f"**File:** `{file_path}`")
                                content_str = (
                                    file_data.get("content", "")
                                    if isinstance(file_data, dict)
                                    else str(file_data)
                                )
                                st.code(content_str[:2000] + ("..." if len(content_str) > 2000 else ""), language="markdown")
                        else:
                            st.caption("No files uploaded to StateBackend in this run.")

                    with tabs[1]:
                        raw_msgs = intermediate.get("raw_messages", [])
                        for rm in raw_msgs:
                            if isinstance(rm, ToolMessage):
                                st.markdown(f"**Tool Response ({rm.name}):**")
                                st.code(str(rm.content)[:2000] + ("..." if len(str(rm.content)) > 2000 else ""), language="markdown")
                            elif hasattr(rm, "tool_calls") and rm.tool_calls:
                                for tc in rm.tool_calls:
                                    st.markdown(f"**Tool Call:** `{tc.get('name')}`")
                                    st.json(tc.get("args", {}))


def process_query(prompt: str):
    """Invoke the deep agent with the given query and render intermediate results."""
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⚡"):
        status_box = st.status("⚡ Agent coordinating workflow...", expanded=True)
        try:
            status_box.write("🔎 Step 1: Searching documentation vector store...")
            agent_instance = st.session_state.agent

            result = agent_instance.invoke(
                {"messages": [HumanMessage(content=prompt)]}
            )

            status_box.write("🧠 Step 2: Analyzing retrieved chunks...")
            status_box.write("✨ Step 3: Synthesizing final answer...")
            status_box.update(label="✅ Answer synthesized successfully!", state="complete", expanded=False)

            # Extract final answer exclusively from AIMessage
            messages = result.get("messages", [])
            final_content = ""

            for m in reversed(messages):
                if isinstance(m, AIMessage):
                    text = extract_message_text(m.content).strip()
                    if text:
                        final_content = text
                        break

            # If model produced only tool calls without text synthesis, synthesize now from tool evidence
            if not final_content:
                status_box.write("📝 Synthesizing answer from retrieved chunks...")
                tool_evidence = []
                for m in messages:
                    if isinstance(m, ToolMessage) and m.content:
                        tool_evidence.append(str(m.content))

                if tool_evidence:
                    from langchain.chat_models import init_chat_model

                    synthesis_llm = init_chat_model(
                        model=st.session_state.selected_model,
                        model_provider="google_genai",
                    )
                    synth_prompt = (
                        f"Based on the following retrieved LangChain/DeepAgents documentation, provide a comprehensive, clear answer to the user's question:\n\n"
                        f"Question: {prompt}\n\n"
                        f"Retrieved Documentation Evidence:\n"
                        + "\n---\n".join(tool_evidence[:2])
                        + "\n\nProvide the complete answer with inline links to documentation sources."
                    )
                    synth_resp = synthesis_llm.invoke(synth_prompt)
                    final_content = extract_message_text(synth_resp.content)

            if not final_content:
                final_content = "The agent retrieved documentation chunks into the virtual filesystem (`/retrieved/`). See the inspection tab below for retrieved content."

            st.markdown(final_content)

            # Store in session state with metadata
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_content,
                    "intermediate": {
                        "files": result.get("files", {}),
                        "raw_messages": messages,
                    },
                }
            )

        except Exception as e:
            err_str = str(e)
            if "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                status_box.update(label="❌ Daily Quota Exceeded", state="error")
                st.error(
                    f"**Daily Free-Tier Quota Limit Reached on `{st.session_state.selected_model}`!**\n\n"
                    "Google's Free Tier has strict daily limits on certain models.\n\n"
                    "👉 **Solution:** Please switch to **`gemini-3.6-flash`** or **`gemini-flash-lite-latest`** in the sidebar dropdown and re-try!"
                )
            else:
                status_box.update(label=f"❌ Error during execution: {e}", state="error")
                st.error(f"Execution failed: {e}")


def run_streamlit_app():
    """Main Streamlit interface controller."""
    init_streamlit_page()
    render_sidebar()
    render_chat_interface()

    # Check if a prompt was selected from quick starters
    prompt_to_run = None
    if st.session_state.selected_query:
        prompt_to_run = st.session_state.selected_query
        st.session_state.selected_query = None

    user_input = st.chat_input("Ask any question about LangChain, DeepAgents, or subagent streaming...")
    if user_input:
        prompt_to_run = user_input

    if prompt_to_run:
        process_query(prompt_to_run)



# CLI EXECUTION IMPLEMENTATION



def run_cli(query: str = EXAMPLE_QUERY, model_name: str = "gemini-flash-lite-latest"):
    """Execute the agent from the command line interface."""
    print("=" * 80)
    print(f"⚡ LangChain DeepAgents RAG System (CLI Mode - Model: {model_name})")
    print("=" * 80)
    print(f"Query: {query}\n")
    print("Invoking agent pipeline...")

    agent_instance = get_agent(model_name=model_name)
    result = agent_instance.invoke(
        {"messages": [HumanMessage(content=query)]}
    )

    print("\n--- Final Agent Response ---")
    final_text = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            t = extract_message_text(msg.content).strip()
            if t:
                final_text = t
                break

    if final_text:
        print(final_text)
    else:
        print("Retrieved files saved. See files below.")

    print("\n" + "=" * 80)
    print(f"Uploaded files in StateBackend: {list(result.get('files', {}).keys())}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        is_streamlit = get_script_run_ctx() is not None
    except Exception:
        is_streamlit = False

    if is_streamlit:
        run_streamlit_app()
    else:
        if len(sys.argv) > 1 and sys.argv[1] == "--cli":
            cli_query = sys.argv[2] if len(sys.argv) > 2 else EXAMPLE_QUERY
            run_cli(cli_query)
        else:
            print("To launch the Streamlit Web Application:")
            print("    streamlit run main.py")
            print("\nRunning example query in CLI mode for verification...\n")
            run_cli(EXAMPLE_QUERY)
