import datetime
import os
import re
import sys
import time
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
from deepagents.backends import StateBackend
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agent import get_agent
from document_processor import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    build_user_vector_store,
    extract_text_from_file,
    validate_file_security,
)
from knowledge_base import CACHE_PATH, DOC_PATHS, DOCS_BASE, get_vector_store

EXAMPLE_QUERY = "How do i stream intermediate tool results from a subagent"

AVAILABLE_MODELS = [
    ("gemini-flash-lite-latest", "⚡ Flash-Lite Latest (Recommended - High Quota)"),
    ("gemini-3.1-flash-lite", "💡 Gemini 3.1 Flash-Lite"),
    ("gemini-3.6-flash", "🚀 Gemini 3.6 Flash"),
    ("gemini-2.5-flash", "⚠️ Gemini 2.5 Flash (Strict Daily Limit)"),
]


def redact_sensitive_tokens(text: str) -> str:
    """Mask potential API keys or secret credentials from output strings."""
    if not isinstance(text, str):
        return text
    # Mask Google AI API keys (AIzaSy...)
    text = re.sub(r"AIzaSy[A-Za-z0-9_\-]{33}", "[REDACTED_API_KEY]", text)
    # Mask general key assignment patterns
    text = re.sub(
        r"""((?:api[_-]?key|secret|token)\s*[:=]\s*['"][A-Za-z0-9_\-]{16,}['"])""",
        r'key="[REDACTED]"',
        text,
        flags=re.IGNORECASE,
    )
    return text


def extract_message_text(content) -> str:
    """Extract readable text from message content, handling string or list-of-dicts."""
    if isinstance(content, str):
        return redact_sensitive_tokens(content)
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return redact_sensitive_tokens("\n".join(parts))
    return redact_sensitive_tokens(str(content)) if content else ""


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

        .security-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.25);
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

        .blue-dot {
            width: 8px;
            height: 8px;
            background-color: #38bdf8;
            border-radius: 50%;
            box-shadow: 0 0 8px #38bdf8;
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

        .doc-chip {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
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
        page_title="DeepAgents | Document RAG",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemini-flash-lite-latest"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_query" not in st.session_state:
        st.session_state.selected_query = None

    # Multi-tenant session isolation: private backend & private vector store per session
    if "session_backend" not in st.session_state:
        st.session_state.session_backend = StateBackend()
    if "user_vector_store" not in st.session_state:
        st.session_state.user_vector_store = None
    if "uploaded_doc_metadata" not in st.session_state:
        st.session_state.uploaded_doc_metadata = []
    if "upload_errors" not in st.session_state:
        st.session_state.upload_errors = []

    # Compile or recompile agent if model or vector store changed
    needs_agent_rebuild = (
        "agent" not in st.session_state
        or st.session_state.get("current_agent_model") != st.session_state.selected_model
        or st.session_state.get("current_agent_has_user_docs") != (st.session_state.user_vector_store is not None)
    )
    if needs_agent_rebuild:
        st.session_state.agent = get_agent(
            model_name=st.session_state.selected_model,
            user_vector_store=st.session_state.user_vector_store,
            agent_backend=st.session_state.session_backend,
        )
        st.session_state.current_agent_model = st.session_state.selected_model
        st.session_state.current_agent_has_user_docs = (st.session_state.user_vector_store is not None)


def handle_document_uploads(uploaded_files):
    """Process, securely validate, extract, and index newly uploaded documents."""
    if not uploaded_files:
        return

    # Check if any new file was added
    existing_names = {doc["filename"] for doc in st.session_state.uploaded_doc_metadata}
    files_to_process = [f for f in uploaded_files if f.name not in existing_names]

    if not files_to_process:
        return

    for file_obj in files_to_process:
        file_bytes = file_obj.getvalue()
        # 1. Multi-layer security validation
        is_valid, safe_name, err_msg = validate_file_security(file_obj.name, file_bytes)
        if not is_valid:
            st.session_state.upload_errors.append(f"Security Alert for '{file_obj.name}': {err_msg}")
            continue

        # Prevent duplicate safe names in session
        if any(doc["filename"] == safe_name for doc in st.session_state.uploaded_doc_metadata):
            continue

        try:
            with st.spinner(f"🔒 Parsing & Indexing '{safe_name}'..."):
                # 2. Safe text extraction
                docs = extract_text_from_file(file_bytes, safe_name)
                if not docs:
                    st.session_state.upload_errors.append(
                        f"Document '{safe_name}' contains no readable text content."
                    )
                    continue

                # 3. Vector indexing into session-isolated store
                st.session_state.user_vector_store, chunk_count = build_user_vector_store(
                    documents=docs,
                    existing_store=st.session_state.user_vector_store,
                )

                # 4. Save metadata for UI display
                st.session_state.uploaded_doc_metadata.append(
                    {
                        "filename": safe_name,
                        "raw_name": file_obj.name,
                        "size_kb": max(1, len(file_bytes) // 1024),
                        "chunk_count": chunk_count,
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "type": os.path.splitext(safe_name)[1].upper()[1:],
                    }
                )

                # Rebuild agent to include user documents tool
                st.session_state.agent = get_agent(
                    model_name=st.session_state.selected_model,
                    user_vector_store=st.session_state.user_vector_store,
                    agent_backend=st.session_state.session_backend,
                )
                st.session_state.current_agent_has_user_docs = True

        except Exception as e:
            st.session_state.upload_errors.append(f"Error processing '{safe_name}': {e}")

    st.rerun()


def render_sidebar():
    """Render configuration, document uploader, security status, and starters in sidebar."""
    with st.sidebar:
        st.markdown("### ⚡ Engine & Security")

        # Security Status Pill
        st.markdown(
            """
            <div style="margin-bottom: 12px;">
                <span class="security-pill">
                    <span class="blue-dot"></span> Session Isolation Active
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
            index=model_options.index(st.session_state.selected_model)
            if st.session_state.selected_model in model_options
            else 0,
            help="Flash-Lite offers high quota; Gemini 3.6 Flash offers superior synthesis.",
        )

        if chosen_model != st.session_state.selected_model:
            st.session_state.selected_model = chosen_model
            with st.spinner(f"Switching agent to {chosen_model}..."):
                st.session_state.agent = get_agent(
                    model_name=chosen_model,
                    user_vector_store=st.session_state.user_vector_store,
                    agent_backend=st.session_state.session_backend,
                )
                st.session_state.current_agent_model = chosen_model
            st.rerun()

        st.markdown("---")
        # ======================================================================
        # DOCUMENT UPLOADER & KNOWLEDGE HUB
        # ======================================================================
        st.markdown("### 📄 Upload Documents")
        st.caption(
            "Upload files to chat directly with your document. Protected with strict extension validation and magic-byte scanning."
        )

        uploaded_files = st.file_uploader(
            "Choose documents (PDF, DOCX, TXT, MD, CSV)",
            type=["pdf", "docx", "txt", "md", "csv"],
            accept_multiple_files=True,
            help=f"Max file size: {MAX_FILE_SIZE_BYTES // (1024*1024)} MB per file. Uploaded docs are kept strictly private in your session.",
        )

        if uploaded_files:
            handle_document_uploads(uploaded_files)

        # Display any security or upload error alerts
        if st.session_state.upload_errors:
            for err in st.session_state.upload_errors:
                st.error(err)
            if st.button("Dismiss Alerts", key="dismiss_alerts"):
                st.session_state.upload_errors = []
                st.rerun()

        # Display active uploaded documents list
        if st.session_state.uploaded_doc_metadata:
            st.markdown("#### 📑 Active Session Documents")
            total_user_chunks = sum(
                doc["chunk_count"] for doc in st.session_state.uploaded_doc_metadata
            )
            st.caption(f"{len(st.session_state.uploaded_doc_metadata)} document(s) indexed ({total_user_chunks} chunks)")

            for idx, doc in enumerate(st.session_state.uploaded_doc_metadata):
                st.markdown(
                    f"""
                    <div class="doc-chip">
                        <div>
                            <strong style="color: #38bdf8;">[{doc['type']}]</strong>
                            <span style="font-size: 0.88rem; color: #f1f5f9;"> {doc['filename']}</span>
                            <div style="font-size: 0.75rem; color: #94a3b8;">
                                {doc['size_kb']} KB • {doc['chunk_count']} chunks • {doc['time']}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.button("🗑️ Clear Uploaded Documents", use_container_width=True):
                st.session_state.user_vector_store = None
                st.session_state.uploaded_doc_metadata = []
                st.session_state.session_backend = StateBackend()
                st.session_state.agent = get_agent(
                    model_name=st.session_state.selected_model,
                    user_vector_store=None,
                    agent_backend=st.session_state.session_backend,
                )
                st.session_state.current_agent_has_user_docs = False
                st.success("Session documents cleared.")
                st.rerun()

        st.markdown("---")

        # System Overview Card
        doc_count = len(st.session_state.uploaded_doc_metadata)
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="metric-label">System Status</div>
                <div style="color: #34d399; font-weight: 600; font-size: 0.95rem;">
                    <span class="pulse-dot"></span> Ready & Operational
                </div>
                <div style="margin-top: 10px;" class="metric-label">User Documents</div>
                <div style="color: {'#38bdf8' if doc_count > 0 else '#64748b'}; font-weight: 600; font-size: 0.9rem;">
                    {f"{doc_count} Active File(s)" if doc_count > 0 else "None Uploaded"}
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
            if st.button("🔄 Re-index Docs", help="Re-fetch docs and rebuild the vector cache"):
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

        if st.session_state.uploaded_doc_metadata:
            st.caption("Prompt ideas tailored for your uploaded documents:")
            sample_prompts = [
                ("Summarize Uploaded Document", "Provide a comprehensive executive summary of my uploaded document."),
                ("Key Points & Takeaways", "What are the most critical takeaways, key metrics, and points in my document?"),
                ("Action Items & Findings", "Extract all action items, decisions, or conclusion points from the document."),
                ("LangChain Streaming Docs", EXAMPLE_QUERY),
            ]
        else:
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
        with st.expander("📚 Official Documentation Corpus", expanded=False):
            st.caption(f"{len(DOC_PATHS)} official LangChain/DeepAgents docs indexed:")
            for p in DOC_PATHS:
                doc_name = p.split("/")[-1].replace("-", " ").title()
                st.markdown(f"- [{doc_name}]({DOCS_BASE}/{p})")


def render_chat_interface():
    """Render header and interactive chat history."""
    has_docs = len(st.session_state.uploaded_doc_metadata) > 0
    header_subtitle = (
        "Chat with your uploaded documents or query the official LangChain & DeepAgents knowledge base."
        if has_docs
        else "Multi-agent documentation assistant powered by Gemini, LangChain vector search, and subagent delegation."
    )
    status_label = "Document Chat Active" if has_docs else "Multi-Agent Active"

    st.markdown(
        f"""
        <div class="hero-container">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h1 class="hero-title">DeepAgents Knowledge & Document RAG</h1>
                    <p class="hero-subtitle">{header_subtitle}</p>
                </div>
                <div>
                    <div class="status-pill">
                        <span class="pulse-dot"></span> {status_label}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # If documents are uploaded, show active document badge banner
    if has_docs:
        doc_names = ", ".join(f"`{d['filename']}`" for d in st.session_state.uploaded_doc_metadata)
        st.info(
            f"📄 **Active Uploaded Documents:** {doc_names} — You can ask questions about your documents now!"
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
                                content_clean = redact_sensitive_tokens(content_str)
                                st.code(
                                    content_clean[:2000] + ("..." if len(content_clean) > 2000 else ""),
                                    language="markdown",
                                )
                        else:
                            st.caption("No files uploaded to StateBackend in this run.")

                    with tabs[1]:
                        raw_msgs = intermediate.get("raw_messages", [])
                        for rm in raw_msgs:
                            if isinstance(rm, ToolMessage):
                                st.markdown(f"**Tool Response ({rm.name}):**")
                                clean_rm = redact_sensitive_tokens(str(rm.content))
                                st.code(
                                    clean_rm[:2000] + ("..." if len(clean_rm) > 2000 else ""),
                                    language="markdown",
                                )
                            elif hasattr(rm, "tool_calls") and rm.tool_calls:
                                for tc in rm.tool_calls:
                                    st.markdown(f"**Tool Call:** `{tc.get('name')}`")
                                    st.json(tc.get("args", {}))


def process_query(prompt: str):
    """Invoke the deep agent with the given query and render intermediate results."""
    # Sanitize user query input against control characters
    prompt = prompt.replace("\x00", "").strip()
    if not prompt:
        return

    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⚡"):
        status_box = st.status("⚡ Agent coordinating workflow...", expanded=True)
        try:
            status_box.write("🔎 Step 1: Searching knowledge base & documents...")
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
                        "Based on the following retrieved document and documentation evidence, provide a comprehensive, clear, and factual answer to the user's question:\n\n"
                        f"Question: {prompt}\n\n"
                        "Retrieved Evidence:\n"
                        + "\n---\n".join(tool_evidence[:3])
                        + "\n\nProvide the complete answer citing source documents or URLs."
                    )
                    synth_resp = synthesis_llm.invoke(synth_prompt)
                    final_content = extract_message_text(synth_resp.content)

            if not final_content:
                final_content = (
                    "The agent retrieved matching excerpts into the virtual filesystem (`/retrieved/`). "
                    "See the inspection tab below for details."
                )

            # Security sanitization on final output
            final_content = redact_sensitive_tokens(final_content)
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

    prompt_to_run = None
    if st.session_state.selected_query:
        prompt_to_run = st.session_state.selected_query
        st.session_state.selected_query = None

    placeholder_text = (
        "Ask a question about your uploaded document or LangChain..."
        if st.session_state.uploaded_doc_metadata
        else "Ask any question about LangChain, DeepAgents, or upload your document in the sidebar..."
    )
    user_input = st.chat_input(placeholder_text)
    if user_input:
        prompt_to_run = user_input

    if prompt_to_run:
        process_query(prompt_to_run)


# ==============================================================================
# CLI EXECUTION IMPLEMENTATION
# ==============================================================================


def run_cli(
    query: str = EXAMPLE_QUERY,
    model_name: str = "gemini-flash-lite-latest",
    doc_path: str | None = None,
):
    """Execute the agent from the command line interface with optional document ingestion."""
    print("=" * 80)
    print(f"⚡ DeepAgents Document RAG System (CLI Mode - Model: {model_name})")
    print("=" * 80)

    user_vector_store = None
    if doc_path:
        if not os.path.exists(doc_path):
            print(f"Error: Document path '{doc_path}' not found.")
            return

        with open(doc_path, "rb") as f:
            file_bytes = f.read()

        is_valid, safe_name, err = validate_file_security(os.path.basename(doc_path), file_bytes)
        if not is_valid:
            print(f"Security Alert: Document rejected: {err}")
            return

        print(f"Ingesting user document: {safe_name}...")
        docs = extract_text_from_file(file_bytes, safe_name)
        user_vector_store, chunk_count = build_user_vector_store(docs)
        print(f"Indexed {chunk_count} chunks from {safe_name} into isolated vector store.\n")

    print(f"Query: {query}\n")
    print("Invoking agent pipeline...")

    agent_backend = StateBackend()
    agent_instance = get_agent(
        model_name=model_name,
        user_vector_store=user_vector_store,
        agent_backend=agent_backend,
    )
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
        print(redact_sensitive_tokens(final_text))
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
            cli_doc = None
            if "--doc" in sys.argv:
                doc_idx = sys.argv.index("--doc")
                if doc_idx + 1 < len(sys.argv):
                    cli_doc = sys.argv[doc_idx + 1]
            run_cli(cli_query, doc_path=cli_doc)
        else:
            print("To launch the Streamlit Web Application:")
            print("    streamlit run main.py")
            print("\nRunning example query in CLI mode for verification...\n")
            run_cli(EXAMPLE_QUERY)
