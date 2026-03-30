import logging
from langgraph.graph import StateGraph, END

from schema import PipelineState, empty_state

from nodes import (
    load_config,
    fetch_candidates,
    llm_triage,
    dispatch_to_categories,     
    process_category,
    llm_summarize,
    format_send
)


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("load_config",      load_config)
    graph.add_node("fetch_candidates", fetch_candidates)
    graph.add_node("llm_triage",       llm_triage)
    graph.add_node("process_category", process_category)
    graph.add_node("llm_summarize",    llm_summarize)
    graph.add_node("format_send",      format_send)

    # Linear edges (sequential)
    graph.set_entry_point("load_config")
    graph.add_edge("load_config",      "fetch_candidates")
    graph.add_edge("fetch_candidates", "llm_triage")

    # Fan-out: conditional edge dispatches parallel Send objects
    # dispatch_to_categories returns a list[Send], each targeting
    # "process_category" with a different category's articles.
    # LangGraph runs all three in parallel and waits for all to finish
    # before moving to llm_summarize.
    graph.add_conditional_edges(
        "llm_triage",
        dispatch_to_categories,
        ["process_category"],       # all Sends target this node
    )

    # Fan-in: all process_category branches converge here
    graph.add_edge("process_category", "llm_summarize")
    graph.add_edge("llm_summarize",    "format_send")
    graph.add_edge("format_send",      END)

    return graph.compile()


def run_pipeline() -> PipelineState:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)s %(message)s",
    )
    return build_graph().invoke(empty_state())