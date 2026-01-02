from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

import json
import datetime
import os
import traceback
import logging
import argparse
import re
from typing import List, Dict, Any
from enum import Enum

llm = None
chat_history = ChatMessageHistory()
vector_store = None
embeddings = None

SUMMARY_FILE = "memory/summary.json"
VECTOR_STORE_PATH = "memory/vector_store"

MINT_PERSONALITY = {
    "greeting": "Hello, I'm Mint.",
    "thinking": "Thinking...",
    "planning": "Planning...",
    "executing": "Executing...",
    "learning": "Learning...",
    "confused": "I need more info",
    "success": "Done",
    "error": "Error",
}

SYSTEM_PROMPT = """
You are Mint, an efficient and helpful AI assistant.

Guidelines:
- Be natural and conversational.
- Detect the user's language and reply in that language.
- Be concise and output ONLY the final answer (no internal thoughts, plans, or analysis).
- If the user asks for step-by-step instructions, provide them only when requested.
"""

prompt = PromptTemplate(
    input_variables=["system", "chat_history", "question", "vector_context"],
    template="""
{system}

Previous relevant conversations:
{vector_context}

Recent conversation history:
{chat_history}

User question:
{question}

Answer:
"""
)

class ToolType(Enum):
    SEARCH = "search"
    MATH = "math"
    CODE = "code"
    MEMORY = "memory"

def parse_tool_call(text: str) -> Dict[str, Any]:
    match = re.search(r'\[TOOL:(\w+):(.+?)\]', text)
    if match:
        return {"type": match.group(1), "content": match.group(2)}
    return None

def execute_tool(tool_type: str, content: str) -> str:
    try:
        if tool_type == "math":
            result = eval(content)
            return f"Math result: {result}"
        elif tool_type == "search":
            return search_vector_memory(content)
        elif tool_type == "code":
            return f"Code: {content}\n(Code execution disabled)"
        elif tool_type == "memory":
            return search_vector_memory(content)
        else:
            return "Unknown tool"
    except Exception as e:
        return f"Tool error: {str(e)}"

# =========================
# VECTOR MEMORY FUNCTIONS
# =========================
def initialize_vector_store(model_name="llama3.2"):
    global vector_store, embeddings
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        if os.path.exists(VECTOR_STORE_PATH):
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            logging.info(f"Loaded existing vector store from {VECTOR_STORE_PATH}")
        else:
            vector_store = None
    except Exception as e:
        logging.warning(f"Vector store initialization warning: {e}")
        vector_store = None

def add_to_vector_memory(question: str, answer: str):
    global vector_store
    try:
        text = f"Q: {question}\nA: {answer}"
        # split into chunks for better retrieval
        chunks = split_text_to_chunks(text, chunk_size=800)
        docs = [Document(page_content=c, metadata={"timestamp": datetime.datetime.utcnow().isoformat(), "type": "conversation"}) for c in chunks]
        if vector_store is None:
            vector_store = FAISS.from_documents(docs, embeddings)
        else:
            vector_store.add_documents(docs)
        if get_vector_count() and get_vector_count() % 50 == 0:
            save_vector_memory()
    except Exception as e:
        logging.warning(f"Failed to add to vector memory: {e}")

def search_vector_memory(query: str, k: int = 3) -> str:
    try:
        if vector_store is None:
            return "No memory"
        results = vector_store.similarity_search(query, k=k)
        if not results:
            return ""
        context = "\n\n---\n".join([r.page_content for r in results])
        return context
    except Exception as e:
        logging.warning(f"Vector search failed: {e}")
        return ""


def get_vector_count() -> int:
    """Return number of items in the vector store using several fallbacks."""
    try:
        return int(getattr(vector_store, "index").ntotal)
    except Exception:
        try:
            return int(getattr(vector_store, "ntotal", 0))
        except Exception:
            try:
                # langchain FAISS docstore fallback
                return len(getattr(vector_store, "docstore")._dict)
            except Exception:
                try:
                    return int(getattr(vector_store, "_index").ntotal)
                except Exception:
                    return 0


def detect_language(text: str) -> str:
    """Very small heuristic language detection.
    Returns 'th' for Thai if Thai characters present, otherwise 'en'.
    If pythainlp is available, prefer it for tokenization.
    """
    if not text:
        return "en"
    # Thai unicode block
    if re.search(r"[\u0E00-\u0E7F]", text):
        return "th"
    return "en"


def split_text_to_chunks(text: str, chunk_size: int = 800) -> List[str]:
    """Split text into chunks. Use Thai tokenizer if available for Thai text."""
    lang = detect_language(text)
    tokens = []
    try:
        if lang == "th":
            try:
                from pythainlp import word_tokenize
                words = word_tokenize(text)
                tokens = words
            except Exception:
                # fallback: split by characters
                tokens = list(text)
        else:
            # simple whitespace split for non-Thai
            tokens = text.split()
    except Exception:
        tokens = text.split()

    chunks = []
    cur = []
    cur_len = 0
    for t in tokens:
        add = (len(t) + 1)
        if cur_len + add > chunk_size and cur:
            chunks.append(" ".join(cur))
            cur = [t]
            cur_len = len(t)
        else:
            cur.append(t)
            cur_len += add
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def save_vector_memory():
    try:
        if vector_store is not None:
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
            vector_store.save_local(VECTOR_STORE_PATH)
            logging.debug("Vector store saved")
    except Exception as e:
        logging.warning(f"Failed to save vector store: {e}")


# =========================
# AUTO-PLANNING AGENT (Think-Plan-Do)
# =========================
class MintAgent:
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def think(self, question: str) -> str:
        analysis = (
            "What is being asked?\n"
            "- Main topic\n"
            "- Core need\n"
            "- Required knowledge\n"
        )
        return analysis

    def plan(self, question: str, analysis: str) -> List[str]:
        plan_prompt = f"""
Based on this question: {question}\n\nAnd this analysis: {analysis}\n\nCreate a short 2-3 step plan.
"""
        try:
            result = self.llm.invoke(plan_prompt)
            steps_text = getattr(result, "text", str(result))
            steps = [s.strip() for s in steps_text.split('\n') if s.strip()]
            return steps[:3]
        except:
            return ["Gather context", "Analyze", "Answer"]

    def do(self, question: str, steps: List[str], context: str, system_prompt: str = None) -> str:
        base = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        execution_prompt = (
            base
            + "\n\nPlan:\n"
            + json.dumps(steps, ensure_ascii=False)
            + "\n\nContext:\n"
            + context
            + f"\n\nQuestion: {question}\n\nAnswer concisely:\n"
        )
        try:
            result = self.llm.invoke(execution_prompt)
            return getattr(result, "text", str(result)).strip()
        except Exception as e:
            return f"Execution failed: {e}"

    def run(self, question: str, vector_context: str = "", system_prompt: str = None) -> str:
        analysis = self.think(question)
        steps = self.plan(question, analysis)
        answer = self.do(question, steps, vector_context, system_prompt=system_prompt)
        return answer

def append_interaction(question, answer, filename):
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "user": question,
        "ai": answer,
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        traceback.print_exc()


def load_history_from_file(filename, limit=100):
    if not os.path.exists(filename):
        return 0

    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception:
        traceback.print_exc()
        return 0

    if limit and len(lines) > limit:
        lines = lines[-limit:]

    count = 0
    for line in lines:
        try:
            obj = json.loads(line)
            if obj.get("user"):
                chat_history.add_user_message(obj["user"])
            if obj.get("ai"):
                chat_history.add_ai_message(obj["ai"])
            count += 1
        except Exception:
            continue
    return count


def get_chat_history_text(max_messages=50, max_chars=3000):
    msgs = list(chat_history.messages or [])
    if max_messages and len(msgs) > max_messages:
        msgs = msgs[-max_messages:]

    text = "\n".join(f"{m.type.capitalize()}: {m.content}" for m in msgs)

    if max_chars and len(text) > max_chars:
        text = text[-max_chars:]

    return text


# =========================
# SUMMARY MEMORY (LONG TERM)
# =========================
def load_summary():
    if not os.path.exists(SUMMARY_FILE):
        return ""
    try:
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("summary", "")
    except Exception:
        return ""


def load_user_profile():
    return {}


def save_user_profile(profile: dict):
    pass


def extract_name(text: str) -> str:
    return None


def get_user_name() -> str:
    return None


def update_summary():
    history_text = get_chat_history_text(max_messages=100, max_chars=5000)
    if not history_text.strip():
        return

    summary_prompt = f"""
Summarize the important facts, preferences, and context from the conversation below.
Keep it short, factual, and useful for future conversations.

Conversation:
{history_text}

Summary:
"""
    try:
        result = llm.invoke(summary_prompt)
        summary_text = getattr(result, "text", str(result))
        os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump({"summary": summary_text}, f, ensure_ascii=False, indent=2)
    except Exception:
        logging.exception("Failed to update summary")


# =========================
# CORE CHAIN (Enhanced with Vector Memory & Agent)
# =========================
agent = None

def run_chain(question, save_file, max_messages, max_chars, use_agent=True):
    if not question.strip():
        return "Please enter a non-empty question."
    # name/profile handling disabled per user request
    vector_context = search_vector_memory(question, k=2)
    history_text = get_chat_history_text(max_messages, max_chars)
    summary_text = load_summary()
    full_system_prompt = SYSTEM_PROMPT
    # Detect language and instruct LLM to reply in that language and only output final answer
    user_lang = detect_language(question)
    if user_lang == "th":
        full_system_prompt = "ตอบเป็นภาษาไทยโดยสั้นและชัดเจน (เฉพาะคำตอบสุดท้าย)\n" + full_system_prompt
    else:
        full_system_prompt = "Respond concisely in the user's language (only final answer)\n" + full_system_prompt
    if summary_text:
        full_system_prompt += f"\n\nMemory Summary:\n{summary_text}"
    try:
        if use_agent and agent:
            answer = agent.run(question, vector_context, system_prompt=full_system_prompt)
        else:
            prompt_text = prompt.format(
                system=full_system_prompt,
                chat_history=history_text,
                question=question,
                vector_context=vector_context,
            )
            raw = llm.invoke(prompt_text)
            answer = getattr(raw, "text", str(raw)).strip()
    except Exception as e:
        logging.exception("LLM invocation failed")
        answer = f"Error: {e}"
    
    # Save interaction and add to vector memory
    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)
    append_interaction(question, answer, save_file)
    add_to_vector_memory(question, answer)
    
    if chat_history.messages and len(chat_history.messages) % 50 == 0:
        update_summary()
        save_vector_memory()
    
    return answer



# =========================
# INIT LLM & AGENT
# =========================
def initialize_llm(model_name):
    global llm, agent
    llm = OllamaLLM(model=model_name)
    agent = MintAgent(llm)
    initialize_vector_store(model_name)
    logging.info(f"Mint agent initialized with {model_name}")


# =========================
# MAIN
# =========================
def main(memory_dir):
    global agent
    
    parser = argparse.ArgumentParser("Mint: AI Chatbot with Vector Memory & Planning")
    parser.add_argument("--history-file", default="chat_history.jsonl")
    parser.add_argument("--max-messages", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--model", default="llama3.3")
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--use-agent", action="store_true", default=True)
    args = parser.parse_args()

    memory_dir = os.path.abspath(memory_dir)
    os.makedirs(memory_dir, exist_ok=True)
    args.history_file = os.path.join(memory_dir, args.history_file)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Minimal startup output
    
    initialize_llm(args.model)
    loaded = load_history_from_file(args.history_file, args.max_messages)
    logging.info(f"Loaded {loaded} previous interactions")
    
    print("Type 'exit' to quit, 'memory' to see vector stats")
    interaction_count = 0
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
            
        if user_input.lower() == "exit":
            save_vector_memory()
            break
        
        if user_input.lower() == "memory":
            if vector_store:
                print(f"Vector Store: {get_vector_count()} memories")
            else:
                print("No memories yet")
            continue

        answer = run_chain(
            user_input,
            args.history_file,
            args.max_messages,
            args.max_chars,
            use_agent=args.use_agent,
        )
        # print only the final answer
        print(answer)
        interaction_count += 1


if __name__ == "__main__":
    main(memory_dir="memory")
