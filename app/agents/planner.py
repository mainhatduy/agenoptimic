from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.states import AgenOptimicState
from app.prompts.prompts import PLANNER_SYSTEM_PROMPT
from app.schemas.outputs import PlannerOutput
from app.utils.llm_factory import get_llm


def _build_human_message(state: AgenOptimicState) -> str:
    model = state.optimization_model
    return (
        f"## Original Problem\n{state.question}\n\n"
        f"## Formal Optimization Model\n"
        f"**Variables:**\n{model.variables}\n\n"
        f"**Constraints:**\n{model.constraints}\n\n"
        f"**Objective Function:**\n{model.objective_function}"
    )


def run_planner(state: AgenOptimicState) -> dict:
    llm = get_llm("planner").with_structured_output(PlannerOutput)

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=_build_human_message(state)),
    ]

    result: PlannerOutput = llm.invoke(messages)

    return {"plan": result.plan}
