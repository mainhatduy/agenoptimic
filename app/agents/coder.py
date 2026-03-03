from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.states import AgenOptimicState
from app.prompts.prompts import CODER_RETRY_SYSTEM_PROMPT, CODER_SYSTEM_PROMPT
from app.schemas.outputs import CoderOutput
from app.utils.llm_factory import get_llm


def _build_human_message(state: AgenOptimicState) -> str:
    model = state.optimization_model
    parts = [
        f"## Formal Optimization Model\n"
        f"**Variables:**\n{model.variables}\n\n"
        f"**Constraints:**\n{model.constraints}\n\n"
        f"**Objective Function:**\n{model.objective_function}\n\n"
        f"## Implementation Plan\n{state.plan}",
    ]

    if state.error and state.retry_count > 0:
        parts.append(
            f"\n\n## Previous Code\n```python\n{state.generated_code}\n```"
            f"\n\n## Error from Previous Attempt\n{state.error}"
        )

    return "\n".join(parts)


def run_coder(state: AgenOptimicState) -> dict:
    is_retry = state.retry_count > 0 and state.error is not None
    system_prompt = CODER_RETRY_SYSTEM_PROMPT if is_retry else CODER_SYSTEM_PROMPT

    llm = get_llm("coder").with_structured_output(CoderOutput)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_human_message(state)),
    ]

    result: CoderOutput = llm.invoke(messages)

    return {
        "generated_code": result.code,
        "error": None,
    }
