# app_langgraph.py
import asyncio
import json
import logging
import os
import time
import uuid
import traceback
from typing import List, Dict, Optional, TypedDict, Literal, Annotated
from datetime import datetime
import numpy as np

import openai
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import pypdf
import docx
from youtube_transcript_api import YouTubeTranscriptApi
from duckduckgo_search import DDGS

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# ============================================================================
# Configuration
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-key-here")
if OPENAI_API_KEY == "your-key-here":
    raise ValueError("Please set OPENAI_API_KEY environment variable")

openai.api_key = OPENAI_API_KEY
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
CHROMA_PERSIST_DIR = "./chroma_db"
GLOBAL_COLLECTION_NAME = "acme_policies"
SESSION_HISTORY_LIMIT = 20

# ============================================================================
# Logging & Telemetry
# ============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("compliance_assistant")

class Telemetry:
    def __init__(self):
        self.log_file = "telemetry.log"
        
    def log(self, data: dict):
        data["timestamp"] = datetime.utcnow().isoformat()
        with open(self.log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
        logger.info(f"Telemetry: {data}")

telemetry = Telemetry()

# ============================================================================
# Vector Store Manager (same as before)
# ============================================================================
class VectorStoreManager:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR, 
            settings=Settings(anonymized_telemetry=False)
        )
        self.global_collection = self.chroma_client.get_or_create_collection(
            name=GLOBAL_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        self.session_collections = {}
        self.youtube_collections = {}
        
    def get_session_collection(self, session_id: str):
        if session_id not in self.session_collections:
            col_name = f"session_{session_id}"
            self.session_collections[session_id] = self.chroma_client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self.session_collections[session_id]
    
    def get_youtube_collection(self, video_id: str):
        if video_id not in self.youtube_collections:
            col_name = f"youtube_{video_id}"
            self.youtube_collections[video_id] = self.chroma_client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self.youtube_collections[video_id]
    
    def add_documents(self, collection, documents: List[Dict], embeddings: List[List[float]]):
        ids = [str(uuid.uuid4()) for _ in documents]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[doc["text"] for doc in documents],
            metadatas=[doc.get("metadata", {}) for doc in documents]
        )
        return ids
    
    def query(self, collection, query_embedding: List[float], top_k: int = 5):
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return (
            results["documents"][0] if results["documents"] else [],
            results["metadatas"][0] if results["metadatas"] else [],
            results["distances"][0] if results["distances"] else []
        )
    
    def delete_collection(self, collection_name: str):
        try:
            self.chroma_client.delete_collection(collection_name)
            return True
        except:
            return False

vector_store = VectorStoreManager()

# ============================================================================
# Document Processing (same as before)
# ============================================================================
def parse_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num + 1}]\n{page_text}"
    except Exception as e:
        logger.error(f"PDF parsing error: {e}")
        raise
    return text

def parse_docx(file_path: str) -> str:
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    except Exception as e:
        logger.error(f"DOCX parsing error: {e}")
        raise
    return text

def parse_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"TXT parsing error: {e}")
        raise

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            last_period = text.rfind('. ', start, end)
            last_question = text.rfind('? ', start, end)
            last_exclamation = text.rfind('! ', start, end)
            last_break = max(last_period, last_question, last_exclamation)
            if last_break > start + chunk_size // 2:
                end = last_break + 2
        chunks.append(text[start:end].strip())
        start = end - overlap if end - overlap > start else end
    return [chunk for chunk in chunks if chunk.strip()]

def embed_texts(texts: List[str]) -> List[List[float]]:
    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

# ============================================================================
# LangGraph State Definition
# ============================================================================
class AgentState(TypedDict):
    messages: List  # list of message objects (LangChain format)
    session_id: str
    tool_calls: List[str]
    error: Optional[str]
    citations: List[Dict]
    final_answer: Optional[str]
    tokens_used: int

# ============================================================================
# Tools (decorated with @tool for LangChain)
# ============================================================================
@tool
def rag_tool(query: str, session_id: str) -> str:
    """Retrieve information from internal policy documents and uploaded documents.
    Use this for questions about ACME policies, regulations, compliance procedures, or internal documents."""
    try:
        # Embed query
        query_embedding = embed_texts([query])[0]
        
        # Query global collection
        global_docs, global_meta, _ = vector_store.query(
            vector_store.global_collection, 
            query_embedding, 
            top_k=3
        )
        
        # Query session collection if exists
        session_docs, session_meta, _ = [], [], []
        if session_id in vector_store.session_collections:
            session_docs, session_meta, _ = vector_store.query(
                vector_store.get_session_collection(session_id), 
                query_embedding, 
                top_k=3
            )
        
        # Combine results
        all_docs = global_docs + session_docs
        all_meta = global_meta + session_meta
        
        if not all_docs:
            return "I couldn't find any relevant information in the policy documents."
        
        # Build context with citations
        context_parts = []
        citations = []
        for i, (doc, meta) in enumerate(zip(all_docs, all_meta)):
            source = meta.get("source", "unknown")
            page = meta.get("page", "")
            context_parts.append(f"[{i+1}] {doc}")
            citations.append({
                "text": doc[:200] + "...",
                "source": source,
                "page": page
            })
        
        context = "\n\n".join(context_parts)
        
        # Prepare LLM prompt
        system_prompt = """You are a compliance assistant for ACME Industries. 
Answer the user's question based ONLY on the provided context. 
Cite sources using [1], [2], etc. exactly as numbered in the context.
If the context does not contain the answer, say "I cannot find this information in the policy documents."
Do not make up information."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0
        )
        
        answer = response.choices[0].message.content
        citations_str = "\n\n**Sources:**\n" + "\n".join([f"- [{i+1}] {c['source']}" for i, c in enumerate(citations)])
        
        return answer + citations_str
        
    except Exception as e:
        logger.error(f"RAG tool error: {e}", exc_info=True)
        return "I encountered an error while searching the documents. Please try again."

@tool
def youtube_tool(url: str, question: str, session_id: str) -> str:
    """Answer questions based on a YouTube video (e.g., compliance training, regulator briefing, vendor explainer).
    Only use when the user provides a YouTube URL."""
    try:
        # Extract video ID
        video_id = None
        if "v=" in url:
            video_id = url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]
        else:
            return "Invalid YouTube URL. Please provide a valid YouTube link."
        
        # Fetch transcript
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                transcript = transcript_list.find_transcript(['en']).fetch()
            except:
                transcript = transcript_list.find_transcript(['en', 'es', 'fr', 'de']).fetch()
            
            full_text = " ".join([entry["text"] for entry in transcript])
            if not full_text.strip():
                return "No transcript available for this video."
        except Exception as e:
            logger.error(f"YouTube transcript error: {e}")
            return "Could not retrieve transcript for this video. The video might be private, have disabled transcripts, or be unavailable."
        
        # Get or create YouTube video collection
        collection = vector_store.get_youtube_collection(video_id)
        
        # Index if not already done
        if collection.count() == 0:
            chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
            if not chunks:
                return "The video transcript could not be processed."
            
            embeddings = embed_texts(chunks)
            docs = [
                {
                    "text": chunk,
                    "metadata": {
                        "source": url,
                        "video_id": video_id,
                        "chunk_index": i
                    }
                }
                for i, chunk in enumerate(chunks)
            ]
            vector_store.add_documents(collection, docs, embeddings)
            logger.info(f"Indexed YouTube video {video_id} with {len(chunks)} chunks")
        
        # Query the collection
        query_embedding = embed_texts([question])[0]
        docs, metas, _ = vector_store.query(collection, query_embedding, top_k=5)
        
        if not docs:
            return "I couldn't find relevant information in the video transcript."
        
        # Build context
        context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])
        
        # Answer using LLM
        system_prompt = """You are a compliance assistant. Answer the user's question based ONLY on the provided YouTube transcript context.
Cite sources using [1], [2], etc. as numbered in the context.
If the answer is not in the transcript, say so."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcript context:\n{context}\n\nQuestion: {question}"}
        ]
        
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0
        )
        
        answer = response.choices[0].message.content
        citations_str = f"\n\n**Sources:**\n- YouTube video: {url}"
        
        return answer + citations_str
        
    except Exception as e:
        logger.error(f"YouTube tool error: {e}", exc_info=True)
        return "I encountered an error while processing the YouTube video. Please try again."

@tool
def web_search_tool(query: str) -> str:
    """Fetch live information from the internet for questions about recent events, external news, regulatory amendments, or information not likely in internal documents."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        
        if not results:
            return "No web search results found."
        
        # Build context
        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(f"[{i+1}] {r['title']}: {r['body']}")
        
        context = "\n".join(context_parts)
        
        # Answer using LLM
        system_prompt = """You are a compliance assistant. Answer the user's question based on the web search results.
Cite sources using [1], [2], etc. as numbered in the context.
If the answer is not in the search results, say so."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Search results:\n{context}\n\nQuestion: {query}"}
        ]
        
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0
        )
        
        answer = response.choices[0].message.content
        citations_str = "\n\n**Sources:**\n" + "\n".join([f"- [{i+1}] {r['href']}" for i, r in enumerate(results)])
        
        return answer + citations_str
        
    except Exception as e:
        logger.error(f"Web search error: {e}", exc_info=True)
        return "I encountered an error while searching the web. Please try again."

# ============================================================================
# LangGraph Nodes
# ============================================================================
# Initialize the LLM with tools
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
tools = [rag_tool, youtube_tool, web_search_tool]
llm_with_tools = llm.bind_tools(tools)

def router_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Node that routes the user message to the appropriate tool or direct answer."""
    messages = state["messages"]
    session_id = state["session_id"]
    
    # Get the last user message
    last_message = messages[-1].content if messages else ""
    
    # Add system prompt for routing
    system_prompt = """You are a compliance assistant router for ACME Industries.
Based on the user's message and conversation history, decide which tool to use:

1. **rag_tool**: For questions about:
   - Internal policies and procedures
   - Regulatory documents (GDPR, DPDP, ISO)
   - Uploaded documents
   - Compliance matters
   - General policy questions

2. **youtube_tool**: ONLY when the user provides a YouTube URL and asks about its content.

3. **web_search_tool**: For questions about:
   - Recent regulatory amendments
   - External news and announcements
   - Current events
   - Information not in internal documents

4. **No tool (direct answer)**: For:
   - Greetings ("hi", "hello")
   - General knowledge questions ("what is GDPR?", "explain compliance")
   - Conversational responses
   - Clarification questions

Respond with a direct answer or call the appropriate tool."""
    
    # Prepare messages for LLM
    llm_messages = [
        SystemMessage(content=system_prompt),
        *messages[-10:]  # Last 10 messages for context
    ]
    
    # Call LLM with tools
    response = llm_with_tools.invoke(llm_messages)
    
    # Update state
    state["messages"].append(response)
    
    # Track tool calls for telemetry
    if response.tool_calls:
        state["tool_calls"] = [tool_call["name"] for tool_call in response.tool_calls]
    else:
        state["tool_calls"] = []
    
    return state

def rag_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Execute RAG tool."""
    session_id = state["session_id"]
    last_message = state["messages"][-1]
    
    # Extract the tool call arguments
    if not last_message.tool_calls:
        state["error"] = "No tool call found in RAG node"
        return state
    
    tool_call = last_message.tool_calls[0]
    args = tool_call.get("args", {})
    query = args.get("query", "")
    
    # Execute RAG
    result = rag_tool.invoke({"query": query, "session_id": session_id})
    
    # Add result as a ToolMessage
    tool_message = ToolMessage(content=result, tool_call_id=tool_call["id"])
    state["messages"].append(tool_message)
    state["final_answer"] = result
    state["tokens_used"] = 0  # Will be updated in telemetry
    
    return state

def youtube_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Execute YouTube tool."""
    session_id = state["session_id"]
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        state["error"] = "No tool call found in YouTube node"
        return state
    
    tool_call = last_message.tool_calls[0]
    args = tool_call.get("args", {})
    url = args.get("url", "")
    question = args.get("question", "")
    
    # Execute YouTube
    result = youtube_tool.invoke({"url": url, "question": question, "session_id": session_id})
    
    # Add result
    tool_message = ToolMessage(content=result, tool_call_id=tool_call["id"])
    state["messages"].append(tool_message)
    state["final_answer"] = result
    state["tokens_used"] = 0
    
    return state

def web_search_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Execute Web Search tool."""
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        state["error"] = "No tool call found in Web Search node"
        return state
    
    tool_call = last_message.tool_calls[0]
    args = tool_call.get("args", {})
    query = args.get("query", "")
    
    # Execute Web Search
    result = web_search_tool.invoke({"query": query})
    
    # Add result
    tool_message = ToolMessage(content=result, tool_call_id=tool_call["id"])
    state["messages"].append(tool_message)
    state["final_answer"] = result
    state["tokens_used"] = 0
    
    return state

def direct_answer_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Handle direct answer (no tool call)."""
    last_message = state["messages"][-1]
    state["final_answer"] = last_message.content
    state["tokens_used"] = 0
    return state

def error_handler_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """Handle errors gracefully."""
    error = state.get("error", "Unknown error")
    state["final_answer"] = f"I encountered an error: {error}. Please try again."
    return state

# ============================================================================
# Build LangGraph
# ============================================================================
def create_agent_graph():
    """Build the LangGraph state graph."""
    
    # Initialize graph with AgentState
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("youtube", youtube_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("error_handler", error_handler_node)
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Conditional routing based on tool calls
    def route_tools(state: AgentState) -> Literal["rag", "youtube", "web_search", "direct_answer", "error_handler"]:
        """Determine which node to go to next based on tool calls."""
        last_message = state["messages"][-1] if state["messages"] else None
        
        if state.get("error"):
            return "error_handler"
        
        if not last_message or not hasattr(last_message, "tool_calls"):
            return "direct_answer"
        
        if not last_message.tool_calls:
            return "direct_answer"
        
        # Get the first tool call
        tool_name = last_message.tool_calls[0]["name"]
        
        if tool_name == "rag_tool":
            return "rag"
        elif tool_name == "youtube_tool":
            return "youtube"
        elif tool_name == "web_search_tool":
            return "web_search"
        else:
            return "direct_answer"
    
    # Add conditional edges from router
    graph.add_conditional_edges(
        "router",
        route_tools,
        {
            "rag": "rag",
            "youtube": "youtube",
            "web_search": "web_search",
            "direct_answer": "direct_answer",
            "error_handler": "error_handler"
        }
    )
    
    # All tool nodes go to END
    graph.add_edge("rag", END)
    graph.add_edge("youtube", END)
    graph.add_edge("web_search", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("error_handler", END)
    
    # Add memory (checkpointing)
    memory = MemorySaver()
    
    # Compile the graph
    return graph.compile(checkpointer=memory)

# Create the agent
agent = create_agent_graph()

# ============================================================================
# FastAPI Application
# ============================================================================
app = FastAPI(
    title="ACME Industries Compliance Assistant (LangGraph)",
    description="Enterprise Policy & Compliance Assistant with LangGraph",
    version="1.0.0"
)

# In-memory session storage for conversation history
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tokens: int
    tool_used: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - handles all queries using LangGraph agent"""
    start_time = time.time()
    session_id = request.session_id
    user_message = request.message
    
    # Initialize state
    state = {
        "messages": [HumanMessage(content=user_message)],
        "session_id": session_id,
        "tool_calls": [],
        "error": None,
        "citations": [],
        "final_answer": None,
        "tokens_used": 0
    }
    
    # Create a thread ID for checkpointing
    config = {"configurable": {"thread_id": f"thread_{session_id}"}}
    
    # Run the agent
    try:
        # Invoke the graph
        result_state = agent.invoke(state, config=config)
        
        # Extract final answer
        final_answer = result_state.get("final_answer", "I'm sorry, I couldn't process your request.")
        tool_used = result_state.get("tool_calls", ["none"])[0] if result_state.get("tool_calls") else "none"
        tokens = result_state.get("tokens_used", 0)
        
        # Update session history (for reference)
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append({"role": "user", "content": user_message})
        sessions[session_id].append({"role": "assistant", "content": final_answer})
        
        # Trim history
        if len(sessions[session_id]) > SESSION_HISTORY_LIMIT * 2:
            sessions[session_id] = sessions[session_id][-SESSION_HISTORY_LIMIT * 2:]
        
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        final_answer = "I encountered an internal error. Please try again later."
        tool_used = "error"
        tokens = 0
        
        # Log error
        telemetry.log({
            "session_id": session_id,
            "user_query": user_message[:200],
            "error": str(e),
            "stack_trace": traceback.format_exc()
        })
    
    total_time = (time.time() - start_time) * 1000
    
    # Telemetry
    telemetry.log({
        "session_id": session_id,
        "user_query": user_message[:200],
        "tool_calls": [tool_used] if tool_used != "none" else [],
        "total_response_time_ms": round(total_time, 2),
        "final_tokens": tokens
    })
    
    return ChatResponse(
        session_id=session_id,
        answer=final_answer,
        tokens=tokens,
        tool_used=tool_used
    )

@app.post("/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload a document for the current session (PDF/DOCX/TXT)"""
    try:
        # Save to temporary file
        temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Parse based on extension
        ext = file.filename.split(".")[-1].lower()
        if ext == "pdf":
            text = parse_pdf(temp_path)
        elif ext == "docx":
            text = parse_docx(temp_path)
        elif ext == "txt":
            text = parse_txt(temp_path)
        else:
            os.remove(temp_path)
            raise HTTPException(400, f"Unsupported file type: {ext}")
        
        # Chunk and embed
        chunks = chunk_text(text)
        if not chunks:
            os.remove(temp_path)
            raise HTTPException(400, "File appears to be empty or unreadable")
        
        embeddings = embed_texts(chunks)
        
        # Add to session collection
        collection = vector_store.get_session_collection(session_id)
        docs = [
            {
                "text": chunk,
                "metadata": {
                    "source": file.filename,
                    "upload_time": datetime.utcnow().isoformat()
                }
            }
            for chunk in chunks
        ]
        vector_store.add_documents(collection, docs, embeddings)
        
        os.remove(temp_path)
        
        return {
            "status": "success",
            "message": f"Document '{file.filename}' indexed for session {session_id}.",
            "chunks": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# ============================================================================
# Startup: Load Global Corpus
# ============================================================================
@app.on_event("startup")
def load_global_corpus():
    """Load all policy documents from the policies directory"""
    policies_dir = "./policies"
    if not os.path.exists(policies_dir):
        os.makedirs(policies_dir)
        logger.info(f"Created policies directory at {policies_dir}")
        # Create sample policy files
        sample_policies = {
            "data_retention.txt": """ACME Industries Data Retention Policy:
- Customer data: 7 years from last interaction
- Employee records: 5 years from termination
- Financial records: 10 years
- Regulatory filings: 12 years
- Email communications: 3 years""",
            "employee_handbook.txt": """ACME Industries Employee Handbook:
- Code of Conduct: All employees must follow ethical guidelines
- Leave Policy: 20 days annual leave, 10 sick days
- Remote Work: Hybrid model, 3 days office per week
- Data Security: No sharing of credentials""",
            "gdpr_compliance.txt": """GDPR Compliance Guidelines:
- Data Subject Rights: Access, rectify, erase, portability
- Consent: Explicit, informed, revocable
- Data Protection Officer: Contact compliance@acme.com
- Breach Notification: Within 72 hours"""
        }
        for filename, content in sample_policies.items():
            filepath = os.path.join(policies_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(f"Created sample policy: {filename}")
    
    # Ingest all files
    total_chunks = 0
    for file in os.listdir(policies_dir):
        path = os.path.join(policies_dir, file)
        if not os.path.isfile(path):
            continue
        try:
            if file.endswith(".pdf"):
                text = parse_pdf(path)
            elif file.endswith(".docx"):
                text = parse_docx(path)
            elif file.endswith(".txt"):
                text = parse_txt(path)
            else:
                continue
            chunks = chunk_text(text)
            if not chunks:
                continue
            embeddings = embed_texts(chunks)
            docs = [{"text": chunk, "metadata": {"source": file}} for chunk in chunks]
            vector_store.add_documents(vector_store.global_collection, docs, embeddings)
            total_chunks += len(chunks)
            logger.info(f"Loaded {file}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    logger.info(f"Global corpus loaded: {total_chunks} total chunks from {policies_dir}")

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_langgraph:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
