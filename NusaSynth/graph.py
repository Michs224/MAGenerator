"""Main NusaSynth LangGraph pipeline.

Graph structure:
  contextualizer → generator → sv_subgraph → lv_subgraph → collect
                     ↑                                        │
                     └────────── (retry if to_retry) ─────────┘

Entry point: contextualizer (first pass only).
Retry loop: collect → generator (skip contextualizer).
End: collect returns empty to_retry.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from NusaSynth.lv_graph import build_lv_subgraph
from NusaSynth.nodes import collect_node, contextualizer_node, generator_node
from NusaSynth.state import BatchState
from NusaSynth.sv_graph import build_sv_subgraph


def route_after_collect(state: BatchState) -> Literal["generator", "__end__"]:
    """Conditional routing after collect: retry → generator, or END."""
    if state.get("to_retry"):
        print(f"Retry: {len(state['to_retry'])} kalimat → generator")
        return "generator"
    return END


def build_pipeline():
    """Build and compile the full NusaSynth pipeline.

    Returns a compiled LangGraph that can be invoked with:
        result = pipeline.invoke({
            "seeds": [...],
            "seed_profile": {...},
            "target_label": "negative",
            "target_lang": "jav",
            "lang_name": "Javanese",
        })
    """
    sv_subgraph = build_sv_subgraph()
    lv_subgraph = build_lv_subgraph()

    builder = StateGraph(BatchState)

    builder.add_node("contextualizer", contextualizer_node)
    builder.add_node("generator", generator_node)
    builder.add_node("sentiment_validator", sv_subgraph)
    builder.add_node("linguistic_validator", lv_subgraph)
    builder.add_node("collect", collect_node)

    builder.add_edge(START, "contextualizer")
    builder.add_edge("contextualizer", "generator")
    builder.add_edge("generator", "sentiment_validator")
    builder.add_edge("sentiment_validator", "linguistic_validator")
    builder.add_edge("linguistic_validator", "collect")

    builder.add_conditional_edges("collect", route_after_collect)

    return builder.compile()


if __name__ == "__main__":
    graph = build_pipeline().get_graph()

    with open("doc/nusasynth_graph.mmd", "w") as f:
        f.write(graph.draw_mermaid())

    with open("doc/nusasynth_graph.png", "wb") as f:
        f.write(graph.draw_mermaid_png())

    print("Saved: doc/nusasynth_graph.mmd, doc/nusasynth_graph.png")
