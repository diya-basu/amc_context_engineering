import time
from typing import Any, Dict

import config
import graph_store
import llm_text_client
import ner_pipeline


def _time_call(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return r, time.perf_counter() - t0


def traditional_rag(query: str, store) -> Dict[str, Any]:
    hits, retrieve_time = _time_call(store.retrieve, query, top_k_children=5)
    context = "\n\n---\n\n".join(
        f"[{h['product_name']} | Page {h['page_num']}]\n{h['parent_text']}" for h in hits)

    prompt = f"""Answer using ONLY the context below. If the answer isn't in the
context, say so explicitly.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    answer, llm_time = _time_call(llm_text_client.call_llm, prompt, model_id=config.CLAUDE_MODEL_RELATIONS)

    docs = [{"name": h["source"], "score": round(h["score"], 2),
             "snippet": h["child_text"][:160].replace("\n", " ")} for h in hits]

    return {
        "mode": "traditional", "query": query,
        "answer": answer or "LLM unavailable — check Vertex/Gemini credentials.",
        "docs": docs,
        "retrieve_time": retrieve_time, "llm_time": llm_time,
        "total_time": retrieve_time + llm_time,
    }


# retrieval.py — hybrid_graphrag, pass product_names from what vector search found

def hybrid_graphrag(query: str, store) -> Dict[str, Any]:
    hits, retrieve_time = _time_call(store.retrieve, query, top_k_children=5)
    vector_context = "\n\n---\n\n".join(
        f"[{h['product_name']} | Page {h['page_num']}]\n{h['parent_text']}" for h in hits)

    product_names = {h["product_name"] for h in hits}
    graph_result, graph_time = _time_call(graph_store.get_subgraph_for_query, query, product_names=product_names, hops=1, limit=40)
    graph_context = "\n".join(
        f"{e['s']} --{e['rel']}--> {e['o']} (confidence={e.get('conf', '?')})"
        for e in graph_result["edges"]) or "(no graph relationships matched this query)"

    prompt = f"""Answer using ONLY the context below. GRAPH CONTEXT gives verified
entity relationships; DOCUMENT CONTEXT gives supporting prose. Cite sources as
[1], [2] etc. If the answer isn't in either, say so explicitly.

GRAPH CONTEXT:
{graph_context}

DOCUMENT CONTEXT:
{vector_context}

QUESTION: {query}

ANSWER:"""
    answer, llm_time = _time_call(llm_text_client.call_llm, prompt, model_id=config.CLAUDE_MODEL_RELATIONS)

    confs = [e.get("conf") for e in graph_result["edges"] if isinstance(e.get("conf"), (int, float))]
    avg_conf = sum(confs) / len(confs) if confs else None
    confidence_label = (
        "no graph signal" if avg_conf is None else
        "high confidence" if avg_conf >= 0.7 else
        "medium confidence" if avg_conf >= 0.4 else "low confidence"
    )

    docs = [{"name": h["source"], "score": round(h["score"], 2),
             "snippet": h["child_text"][:160].replace("\n", " ")} for h in hits]

    matched_entities = ner_pipeline.run_layers_ab(query)
    matched_texts = {e["text"] for e in matched_entities}
    active_labels = {e["label"] for e in matched_entities}

    return {
        "mode": "hybrid", "query": query,
        "answer": answer or "LLM unavailable — check Vertex/Gemini credentials.",
        "docs": docs, "confidence_label": confidence_label,
        "graph_nodes": graph_result["nodes"], "graph_edges": graph_result["edges"],
        "matched_entity_texts": matched_texts, "active_labels": active_labels,
        "retrieve_time": retrieve_time, "graph_time": graph_time, "llm_time": llm_time,
        "total_time": retrieve_time + graph_time + llm_time,
    }


def relevancy_score(result: Dict[str, Any]) -> float:
    """Directional proxy metric, not a real eval: citation density + graph
    signal presence + absence of a hedge phrase + answer substance."""
    answer = result.get("answer", "")
    score = 0.0
    score += min(answer.count("[") * 0.15, 0.45)
    score += 0.25 if result.get("graph_edges") else 0
    score += 0.2 if "isn't in" not in answer.lower() and "not in the context" not in answer.lower() else 0
    score += 0.1 if len(answer) > 120 else 0
    return round(min(score, 1.0), 2)