"""
graph_store.py
===============
Neo4j client + schema + MERGE writer. Entities dedupe on ISIN where present,
otherwise on (label, normalized text).
"""
from __future__ import annotations
from typing import Any, Dict, List
import re

import config

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        _driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            max_connection_lifetime=200,       # recycle connections before Aura drops them
            connection_timeout=30,
            max_transaction_retry_time=30,
        )
    return _driver


def init_schema():
    with get_driver().session(database=config.NEO4J_DATABASE) as session:
        session.run("CREATE CONSTRAINT entity_key IF NOT EXISTS "
                     "FOR (e:Entity) REQUIRE e.dedup_key IS UNIQUE")
        session.run("CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)")


def _dedup_key(label: str, text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return f"{label}::{norm}"


def upsert_entities(entities: List[Dict[str, Any]], product_name: str, source: str,
                     batch_size: int = 200):
    """Batched via UNWIND — avoids hundreds of individual round-trips to Aura."""
    driver = get_driver()
    isin_lookup = {e["parent_id"]: e["text"] for e in entities if e["label"] == "ISIN"}

    rows = []
    for e in entities:
        key = e["text"] if e["label"] == "ISIN" else _dedup_key(e["label"], e["text"])
        rows.append({
            "key": key, "text": e["text"], "label": e["label"],
            "isin": isin_lookup.get(e["parent_id"]),
            "product_name": product_name, "source": source,
            "chunk_id": e.get("child_id", ""),
        })

    with driver.session(database=config.NEO4J_DATABASE) as session:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            session.run(
                """
                UNWIND $rows AS row
                MERGE (n:Entity {dedup_key: row.key})
                ON CREATE SET n.text = row.text, n.label = row.label, n.isin = row.isin,
                              n.product_name = row.product_name, n.source = row.source,
                              n.first_seen_chunk = row.chunk_id
                ON MATCH SET n.isin = coalesce(n.isin, row.isin)
                """,
                rows=batch,
            )


def upsert_relations(relations: List[Dict[str, Any]], product_name: str, source: str,
                      batch_size: int = 100):
    """Batched via UNWIND, grouped by predicate (since predicate is used as a
    relationship TYPE and can't be parameterized in Cypher)."""
    driver = get_driver()

    by_predicate: Dict[str, list] = {}
    for r in relations:
        predicate = re.sub(r"[^a-zA-Z_]", "_", r["predicate"].upper())
        by_predicate.setdefault(predicate, []).append({
            "subj_key": _dedup_key("GENERIC", r["subject"]), "subj_text": r["subject"],
            "obj_key": _dedup_key("GENERIC", r["object"]), "obj_text": r["object"],
            "conf": r["confidence"], "chunk_id": r["source_chunk_id"],
            "product_name": product_name, "source": source,
        })

    with driver.session(database=config.NEO4J_DATABASE) as session:
        for predicate, rows in by_predicate.items():
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(
                    f"""
                    UNWIND $rows AS row
                    MERGE (s:Entity {{dedup_key: row.subj_key}})
                      ON CREATE SET s.text = row.subj_text, s.label = 'UNRESOLVED',
                                    s.product_name = row.product_name, s.source = row.source
                    MERGE (o:Entity {{dedup_key: row.obj_key}})
                      ON CREATE SET o.text = row.obj_text, o.label = 'UNRESOLVED',
                                    o.product_name = row.product_name, o.source = row.source
                    MERGE (s)-[rel:{predicate}]->(o)
                      ON CREATE SET rel.confidence = row.conf, rel.source_chunk_id = row.chunk_id
                      ON MATCH SET rel.confidence = CASE WHEN row.conf > rel.confidence
                                                          THEN row.conf ELSE rel.confidence END
                    """,
                    rows=batch,
                )

def resolve_unresolved_entities():
    """
    Post-pass: for UNRESOLVED nodes created by relation MERGE (because the
    subject/object text wasn't seen by NER Layer A/B directly), try to match
    them onto an existing resolved Entity node with the same normalized text.
    """
    with get_driver().session(database=config.NEO4J_DATABASE) as session:
        session.run("""
            MATCH (u:Entity {label: 'UNRESOLVED'})
            MATCH (r:Entity) WHERE r.label <> 'UNRESOLVED'
              AND toLower(trim(r.text)) = toLower(trim(u.text))
            SET u.label = r.label, u.isin = coalesce(u.isin, r.isin)
        """)


def get_subgraph(limit: int = 200) -> Dict[str, list]:
    with get_driver().session(database=config.NEO4J_DATABASE) as session:
        result = session.run(
            "MATCH (s:Entity)-[r]->(o:Entity) "
            "RETURN s.text AS s, s.label AS s_label, type(r) AS rel, "
            "       r.confidence AS conf, o.text AS o, o.label AS o_label "
            "LIMIT $limit", limit=limit)
        rows = [dict(record) for record in result]
    return {"edges": rows}

# graph_store.py — replace get_subgraph_for_query

def get_subgraph_for_query(query: str, product_names: set | None = None,
                            hops: int = 1, limit: int = 40) -> Dict[str, list]:
    import ner_pipeline
    hops = max(int(hops), 1)
    query_entities = ner_pipeline.run_layers_ab(query)
    entity_texts = list({e["text"] for e in query_entities})

    edges = []

    # Path 1 — literal entity mentions in the query text (works for "who manages Axis X")
    if entity_texts:
        with get_driver().session(database=config.NEO4J_DATABASE) as session:
            result = session.run(
                f"""
                MATCH (n:Entity) WHERE n.text IN $texts
                MATCH path = (n)-[r*1..{hops}]-(m)
                UNWIND relationships(path) AS rel
                WITH startNode(rel) AS s, rel, endNode(rel) AS o
                RETURN DISTINCT s.text AS s, s.label AS s_label, type(rel) AS rel,
                       rel.confidence AS conf, o.text AS o, o.label AS o_label
                LIMIT $limit
                """, texts=entity_texts, limit=limit)
            edges += [dict(r) for r in result]

    # Path 2 — always fall back to the product(s) vector search already found.
    # This is what guarantees a graph shows up even for conceptual questions
    # that never name the scheme literally.
    if not edges and product_names:
        with get_driver().session(database=config.NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (n:Entity) WHERE n.product_name IN $products
                MATCH (n)-[r]-(m)
                RETURN DISTINCT n.text AS s, n.label AS s_label, type(r) AS rel,
                       r.confidence AS conf, m.text AS o, m.label AS o_label
                LIMIT $limit
                """, products=list(product_names), limit=limit)
            edges += [dict(r) for r in result]

    # Path 3 — last resort, so the panel is never structurally empty
    if not edges:
        with get_driver().session(database=config.NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (n:Entity)-[r]-(m)
                WITH n, r, m, size((n)--()) AS degree
                ORDER BY degree DESC LIMIT $limit
                RETURN n.text AS s, n.label AS s_label, type(r) AS rel,
                       r.confidence AS conf, m.text AS o, m.label AS o_label
                """, limit=limit)
            edges += [dict(r) for r in result]

    nodes = list({e["s"] for e in edges} | {e["o"] for e in edges})
    return {"nodes": nodes, "edges": edges, "matched_by": "entity" if entity_texts else
            ("product" if product_names else "fallback")}

def get_entity_type_summary(active_labels: set | None = None) -> list[dict]:
    """Real counts per entity label — powers the Ontology View tree.
    active_labels marks which types this specific query's matched entities belong to."""
    with get_driver().session(database=config.NEO4J_DATABASE) as session:
        result = session.run(
            "MATCH (n:Entity) RETURN n.label AS label, count(*) AS n ORDER BY n DESC")
        rows = [dict(r) for r in result]
    active_labels = active_labels or set()
    return [{"label": r["label"] or "UNLABELED", "count": r["n"],
              "active": r["label"] in active_labels} for r in rows]