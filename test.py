from langchain_classic.prompts import PromptTemplate
from .vectordb import search_store

INSUFFICIENT = "Insufficient Information"

DOMAIN_KEYWORDS = {
    "vulnerability","patch","phishing","intrusion","ransomware","malware","incident",
}

PERMISSIONS = {
    "planner": ["plan"],
    "worker": ["retrieve", "answer"],
    "synthesizer": ["synthesize"],
}

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=('''
        Answer ONLY using the provided legal context.
        If the answer is not present, respond with:
        Insufficient Information
        Context:
        {context}
        Question:
        {question}
    '''
    )
)

def build_prompt(context, question):
    return RAG_PROMPT.format(context=context,question=question)

def get_answer(chunks, question):
    q_words = set(question.lower().split())
    best, best_score = INSUFFICIENT, -1
    for chunk in chunks:
        content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
        for sentence in content.replace("\n", ". ").split("."):
            s = sentence.strip()
            score = len(q_words & set(s.lower().split()))
            if score > best_score:
                best_score, best = score, s
    return best

def rag_answer(collection, question, top_k=3):
    hits = search_store(collection, question, top_k=top_k)

    return {
        "question": question,
        "answer": hits[0]["content"] if hits else INSUFFICIENT,
        "retrieved_sources": sorted(
            {h["metadata"]["source"] for h in hits if "source" in h["metadata"]}
        ),
        "confidence_score": hits[0]["score"] if hits else 0.0
    }

def needs_retrieval(question):
    return bool(set(question.lower().split()) & DOMAIN_KEYWORDS)

def agentic_answer(collection, question):
    if not needs_retrieval(question):
        return {
            "question": question,
            "decision": "direct",
            "answer": "Direct response not requiring retrieval.",
            "sources": [],
            "grounded": False,
        }

    rag = rag_answer(collection, question)

    return {
        "question": question,
        "decision": "retrieve",
        "answer": rag["answer"],
        "sources": rag["retrieved_sources"],
        "grounded": rag["confidence_score"] >= 0.5,
    }

def planner(question):
    return [p for p in question.replace("?", " and ").split(" and ") if p]

def worker(collection,sub_query):
    if not any(w in DOMAIN_KEYWORDS for w in sub_query.lower().split()):
        return {
            "sub_query":sub_query,
            "answer":"Please ask about company policies.",
            "sources":[]
        }
    r=rag_answer(collection,sub_query)
    return {
        "sub_query":sub_query,
        "answer":r["answer"],
        "sources":r["retrieved_sources"]
    }

def synthesizer(worker_results):
    answers = [w["answer"] for w in worker_results if w.get("answer")]
    sources = sorted({s for w in worker_results for s in w.get("sources", []) if s})
    return {
        "final_answer": " ".join(answers) if answers else INSUFFICIENT,
        "sources":sources
    }

def is_allowed(agent,action):
    return action in PERMISSIONS.get(agent,[])

def validate_answer(answer,context):
    if not context:
        return {"is_grounded":False,"confidence":0.0}
    a=answer.lower().split()
    c=context.lower().split()
    conf=sum(w in c for w in a)/len(a)
    return {
        "is_grounded":conf>=0.5,
        "confidence":conf
    }

def run_workflow(collection, question: str):
    trace = []
    message_log = []

    # Step 1 — planner
    sub_queries = planner(question)
    trace.append({"node": "planner", "sub_queries": sub_queries})

    # Step 2 — workers
    worker_results = []
    for sq in sub_queries:
        message_log.append({"from": "planner", "to": "worker", "content": sq})
        result = worker(collection, sq)
        worker_results.append(result)
        trace.append({"node": "worker", "sub_query": sq})
        message_log.append({"from": "worker", "to": "synthesizer", "content": result["answer"]})

    # Step 3 — synthesizer
    synth = synthesizer(worker_results)
    trace.append({"node": "synthesizer"})

    # Step 4 — validate (inline word overlap)
    context    = " ".join(w.get("answer", "") for w in worker_results)
    a_words    = set(synth["final_answer"].lower().split())
    c_words    = set(context.lower().split())
    confidence = round(len(a_words & c_words) / len(a_words), 2) if a_words else 0.0
    validation = {"is_grounded": confidence >= 0.5, "confidence": min(confidence, 1.0)}

    return {
        "question": question,
        "final_answer": synth["final_answer"],
        "sources": synth["sources"],
        "trace": trace,
        "message_log": message_log,
        "validation": validation
    }
    

# ==================================================================================================
# api.py

"""api.py — Task 9: FastAPI REST API (ThreatIntel).

One shared in-memory store is built at import and reused by every endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .loader import load_documents
from .chunking import chunk_documents
from .vectordb import build_store
from .agents import rag_answer, run_workflow

app = FastAPI(title="ThreatIntel API")

def build_collection(data_dir="security_docs"):
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)
    return build_store(chunks)

collection = build_collection()

class IngestRequest(BaseModel):
    data_dir: str

class QuestionRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {
        "status": "running",
        "indexed": collection.count()
    }

@app.post("/ingest")
def ingest(req: IngestRequest):
    global collection

    docs = load_documents(req.data_dir)
    chunks = chunk_documents(docs)

    if len(chunks) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents found"
        )

    collection = build_store(chunks)

    return {
        "chunks_indexed": collection.count()
    }

@app.post("/ask")
def ask(req: QuestionRequest):

    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    return rag_answer(
        collection,
        req.question
    )

@app.post("/workflow")
def workflow(req: QuestionRequest):

    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    return run_workflow(
        collection,
        req.question
    )

