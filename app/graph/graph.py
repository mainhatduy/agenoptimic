from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agents.coder import run_coder
from app.agents.formulator import run_formulator
from app.agents.planner import run_planner
from app.graph.states import AgenOptimicState
from app.tools.code_executor import execute_python_code
from app.utils.llm_factory import get_settings


def run_executor(state: AgenOptimicState) -> dict:
    result = execute_python_code(state.generated_code)

    if result.success:
        return {
            "solution": result.stdout.strip(),
            "error": None,
        }
    else:
        return {
            "solution": None,
            "error": result.stderr.strip(),
            "retry_count": state.retry_count + 1,
        }


def should_retry(state: AgenOptimicState) -> Literal["coder", "__end__"]:
    settings = get_settings()
    if state.error and state.retry_count <= settings.max_retries:
        return "coder"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(AgenOptimicState)

    graph.add_node("formulator", run_formulator)
    graph.add_node("planner", run_planner)
    graph.add_node("coder", run_coder)
    graph.add_node("executor", run_executor)

    graph.add_edge(START, "formulator")
    graph.add_edge("formulator", "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "executor")
    graph.add_conditional_edges("executor", should_retry)

    return graph


compiled_graph = build_graph().compile()
