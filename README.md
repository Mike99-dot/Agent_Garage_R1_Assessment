# Agent_Garage_R1_Assessment (LangGraph)

An Enterprise Policy & Compliance Assistant built with FastAPI and LangGraph that helps employees navigate internal policies, regulatory guidance, and workplace queries through natural-language conversation. The system uses a multi‑tool agent architecture with RAG, YouTube Q&A, web search, and telemetry.

## 🚀 Features
Document‑grounded Q&A (RAG) – Answers from an ingested corpus of PDF/DOCX/TXT policy documents; supports mid‑conversation uploads without re‑indexing.

  - YouTube Q&A – Extracts and indexes video transcripts (persistent vector store) to answer questions about compliance training or regulatory briefings.

  - Web Search – Fetches live external information (e.g., regulatory amendments, news) via DuckDuckGo.

  - Multi‑turn conversation – Retains conversation history per session; summarises and trims long histories.

  - General knowledge – Handles greetings, explanations, and conversational queries without invoking tools.

  - Resilience – Graceful fallback on LLM or tool failures; error handler node in the LangGraph.

  - Telemetry – Logs token consumption, response time, tool calls, and stack traces for every request.

  - Document ingestion – Supports PDF, DOCX, and TXT with chunking and OpenAI embeddings.

### # 🏗️ Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────┐
│  FastAPI (app)                                             │
│  ├─ /chat                                                  │
│  ├─ /upload                                                │
│  └─ /health                                                │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│  LangGraph Agent (StateGraph)                              │
│  ├─ State: messages, session_id, tool_calls, errors        │
│  ├─ Nodes: router, rag, youtube, web_search, direct_answer │
│  ├─ Conditional edges based on tool calls                  │
│  └─ Checkpointer: MemorySaver (session persistence)        │
└───────┬──────────────┬──────────────┬──────────────┬───────┘
        │              │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
│ RAG Tool     │ │ YouTube    │ │ Web Search │ │ General    │
│ - Chroma DB  │ │ Tool       │ │ Tool       │ │ Knowledge  │
│ - Global &   │ │ - Vector DB│ │ - DuckDuckGo│ │ (LLM only)│
│   session    │ │ - Persistent│ │ - Synthesis │ │           │
│   collections│ │ - Citations │ │ - Citations │ │           │
└──────────────┘ └─────────────┘ └─────────────┘ └───────────┘
        │              │              │              │
┌───────▼─────────────────────────────────────────────────────┐
│  Telemetry & Logging                                       │
│  - Token consumption, response time, tool calls, errors    │
└─────────────────────────────────────────────────────────────┘
```

# Compliance Assistant

A FastAPI + LangGraph powered compliance assistant with support for:

- Retrieval-Augmented Generation (RAG)
- YouTube transcript Q&A
- Web search integration
- Session memory
- Document upload and indexing
- Telemetry logging

Built with FastAPI, LangGraph, OpenAI, ChromaDB, YouTube Transcript API, and DuckDuckGo Search.

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-directory>
```

## 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Set your OpenAI API key

### Linux/macOS

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### Windows (PowerShell)

```powershell
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

### Optional `.env` file

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

## 5. Start the server

```bash
python app_langgraph.py
```

The server will run at:

```text
http://localhost:8000
```

On startup, the application creates a `./policies` directory with sample policy files and indexes them automatically.

---

# 📘 API Reference

## POST `/chat`

Handles a user query and returns the assistant's answer.

### Request Body

```json
{
  "session_id": "alice",
  "message": "What is our data retention policy?"
}
```

### Response

```json
{
  "session_id": "alice",
  "answer": "ACME's data retention policy requires customer data to be kept for 7 years...\n\n**Sources:**\n- [1] data_retention.txt",
  "tokens": 670,
  "tool_used": "rag_tool"
}
```

---

## POST `/upload`

Uploads a document (`PDF`, `DOCX`, or `TXT`) and indexes it for the session.

### Request

Multipart form data with:
- `session_id`
- `file`

### Example

```bash
curl -X POST "http://localhost:8000/upload?session_id=alice" \
  -F "file=@./leave_policy.txt"
```

### Response

```json
{
  "status": "success",
  "message": "Document 'leave_policy.txt' indexed for session alice.",
  "chunks": 12
}
```

---

## GET `/health`

Health check endpoint.

### Response

```json
{
  "status": "healthy",
  "timestamp": "2025-05-22T10:30:00Z"
}
```

---

# 🧪 Testing Examples

## 1. General Conversation

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "message": "Hi, what is GDPR?"
  }'
```

---

## 2. RAG (Policy Query)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "message": "What is our data retention policy for customer data?"
  }'
```

---

## 3. Uploaded Document Query

### Upload Document

```bash
curl -X POST "http://localhost:8000/upload?session_id=test" \
  -F "file=@./leave_policy.txt"
```

### Ask Questions About the Uploaded Document

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "message": "What does the leave policy say about sick days?"
  }'
```

---

## 4. YouTube Q&A

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "message": "What does this video say about compliance? https://www.youtube.com/watch?v=VIDEO_ID"
  }'
```

---

## 5. Web Search

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "message": "What is the latest SEBI announcement about ESG?"
  }'
```

---

## 6. Multi-turn Conversation

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "alice",
    "message": "What is our retention policy for HR records?"
  }'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "alice",
    "message": "And what about for EU employees?"
  }'
```

---

# 📊 Telemetry

All requests are logged to `telemetry.log` in JSON Lines format.

## Example Log Entry

```json
{
  "timestamp": "2025-05-22T10:32:15.123Z",
  "session_id": "alice",
  "user_query": "What is our data retention policy?",
  "tool_calls": ["rag_tool"],
  "input_tokens": 450,
  "output_tokens": 120,
  "total_tokens": 570,
  "total_response_time_ms": 1245,
  "final_tokens": 570
}
```

Error logs additionally include:
- `error`
- `stack_trace`

---

# 🛠️ LangGraph Implementation Details

## StateGraph

Defines the workflow nodes:
- `router`
- `rag`
- `youtube`
- `web_search`
- `direct_answer`
- `error_handler`

Conditional edges determine which node executes next.

---

## Tools

Tools are decorated with `@tool` for:
- Automatic schema generation
- Tool binding with the LLM

---

## Routing

The `router_node` uses:

```python
llm.bind_tools()
```

to determine which tool should handle the query.

---

## Checkpointing

Uses `MemorySaver` for thread-based session persistence.

Conversation state is retained across requests using the same `session_id`.

---

## Error Handling

A dedicated `error_handler_node` catches failures and returns graceful responses.

---

# 🐳 Docker Deployment (Optional)

## Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app_langgraph:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Build Docker Image

```bash
docker build -t compliance-assistant .
```

---

## Run Docker Container

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-xxx \
  compliance-assistant
```

---

# 📁 Project Structure

```text
.
├── app_langgraph.py
├── requirements.txt
├── telemetry.log
├── policies/
├── uploads/
└── README.md
```

---

# 🔧 Technologies Used

- FastAPI
- LangGraph
- OpenAI API
- ChromaDB
- YouTube Transcript API
- DuckDuckGo Search

---

# 📄 License

This project is for demonstration purposes as part of a technical assessment.
