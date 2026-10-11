# DeepAgents Knowledge & Document RAG

A production-grade Retrieval-Augmented Generation (RAG) assistant powered by Google Gemini, LangChain, and DeepAgents. Features multi-source retrieval over official technical documentation and user-uploaded documents with enterprise-grade security and multi-tenant session isolation.

---

## 🚀 Features

- **📄 Document Uploader**: Upload and chat with your own documents (`.pdf`, `.docx`, `.txt`, `.md`, `.csv`).
- **🛡️ Enterprise-Grade Security**:
  - **Multi-tenant session isolation**: User-uploaded documents and in-memory vector stores are strictly isolated per browser session in `st.session_state` and never leaked across sessions or written to shared disk caches.
  - **Path traversal prevention & filename sanitization**: Raw file paths and malicious traversal sequences (`../`, `..\`) are stripped.
  - **Magic-byte file header inspection**: Validates actual binary signatures (`%PDF-`, `PK\x03\x04`) and blocks executable spoofing (`.exe`, `.elf`, scripts).
  - **Resource limits (Anti-DoS)**: Max 10MB per file, max 100 pages per PDF, and max 250 chunks per document to protect memory and API quotas.
  - **Indirect prompt injection defense**: Document excerpts are wrapped in strict structural delimiters (`<<<UNTRUSTED_USER_DOCUMENT_CHUNK>>>`), with explicit system instructions prohibiting execution of untrusted commands.
  - **Secret token scrubber**: Automatic regex scrubbing to redact accidental leakage of API keys or credentials in outputs.
- **⚡ Dual Source Retrieval**: Automatically queries user documents for user data and official LangChain/DeepAgents docs for framework questions.
- **🔍 Inspection & Transparency**: Live inspection tabs for retrieved chunks, subagent activity, and StateBackend virtual files.
- **✨ Glassmorphic UI**: Modern dark theme interface with Outfit typography, status indicators, and prompt starters.

---

## Supported File Formats

| Format        | Extension | Processing Engine   | Security Checks                                         |
| ------------- | --------- | ------------------- | ------------------------------------------------------- |
| PDF           | `.pdf`    | `pypdf`             | `%PDF-` magic header, encryption check, max 100 pages   |
| Word Document | `.docx`   | `python-docx`       | `PK\x03\x04` zip header check, table & text extraction  |
| Plain Text    | `.txt`    | Safe UTF-8/Latin-1  | Null byte check, binary signature blocking              |
| Markdown      | `.md`     | Safe UTF-8/Latin-1  | Null byte check, binary signature blocking              |
| Tabular Data  | `.csv`    | Python `csv.reader` | Row limits (max 2,500 rows), structured markdown tables |

---
