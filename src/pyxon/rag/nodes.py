# src.pyxon.rag.nodes
import json
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from src.pyxon.rag.schemas import ReflectionDecision
from src.pyxon.rag.utils import format_docs_summary, format_previous_critiques, format_queries_history
from src.pyxon.retrieval.reranker import CrossEncoderReranker
from src.pyxon.storage.vs import VectorStore

from src.config import Settings
from src.pyxon.rag.prompts import (
    GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    REFLECTION_PROMPT,
)
from src.pyxon.rag.state import AgentState

_vs = VectorStore()._vs
_reranker = CrossEncoderReranker()
_llm = ChatGroq(model=Settings.LLM_MODEL_NAME)

def retrieve(state: AgentState) -> AgentState:
    if len(state["queries"]) == 0:
        state["queries"].append(state['messages'][-1].content)

    query = state["queries"][-1]
    similarity_threshold = Settings.SIMILARITY_THRESHOLD
    metadata_filter = state.get("metadata_filter")

    search_kwargs = {"k": 20, "similarity_score_threshold": similarity_threshold}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retrieved_docs = _vs.search(
        query=query, 
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs
    )
    
    state["retrieved_docs"] = retrieved_docs

    return state

tst = AgentState(
    messages=[HumanMessage(content="What is the role of Ahmad Alqaisi in Cyber robot company?")],
    queries=[], 
    critiques=[],                    
    iteration=0,                     
    should_continue=False,           
    retrieved_docs=[],               
    reranked_docs=[],                
    top_k=5,                         
    similarity_score_threshold=0.5,  
    metadata_filter=None,            
    answer=""                        
)

def rerank(state: AgentState) -> AgentState:
    top_k = state.get("top_k", Settings.TOP_K)
    query = state["queries"][-1]

    reranked_docs = _reranker.rerank(
        query=query, documents=state["retrieved_docs"], top_k=top_k
    )
    state["reranked_docs"] = reranked_docs

    return state


def reflect(state: AgentState) -> AgentState:
    queries = state.get("queries", [])
    critiques = state.get("critiques", [])
    docs = state.get("reranked_docs", [])
    current_top_k = state.get("top_k", Settings.TOP_K)
    current_filter = state.get("metadata_filter", None)
    
    prompt_input = {
        "original_q": state['messages'][-1].content,
        "queries_history":      format_queries_history(queries),
        "previous_critiques":   format_previous_critiques(critiques),
        "current_docs_summary": format_docs_summary(docs),
        "current_top_k": current_top_k,
        "current_filter": json.dumps(current_filter) if current_filter else "null",
        "iteration_count": state['iteration']
    }

    llm = _llm.with_structured_output(ReflectionDecision)
    chain = REFLECTION_PROMPT | llm 
    response: ReflectionDecision = chain.invoke(prompt_input)

    if response.should_continue:

        state["critiques"] = critiques + [response.critique]
        state["should_continue"] = response.should_continue
        state["top_k"] = response.top_k
        
        if response.filter and response.filter.get("document_id"):
            try:
                UUID(response.filter["document_id"])
                state["metadata_filter"] = response.filter
            except ValueError:
                state["metadata_filter"] = current_filter
        else:
            state["metadata_filter"] = None
            
    else:
        state["should_continue"] = False
    
    return state


def rewrite_query(state: AgentState) -> AgentState:
    original_question = state['messages'][-1].content
    latest_critique = state["critiques"][-1]

    previous_queries = format_queries_history(state["queries"])
    previous_critiques = format_previous_critiques(state["critiques"])

    rewrite_prompt = QUERY_REWRITE_PROMPT.format(
        original_question=original_question,
        latest_critique=latest_critique,
        previous_queries=previous_queries,
        previous_critiques=previous_critiques,
    )

    new_query = _llm.invoke(rewrite_prompt).content.strip()
    state["queries"].append(new_query)

    return state


def generate(state: AgentState) -> AgentState:
    question = state["messages"][-1].content if state["messages"] else ""

    if state["reranked_docs"]:
        context = format_docs_summary(state['reranked_docs'])
    else:
        context = "No relevant documents found."

    prompt = GENERATION_PROMPT.format(context=context, input=question)
    answer = _llm.invoke(prompt).content

    state["messages"].append(AIMessage(content=answer))
    state["answer"] = answer

    return state
