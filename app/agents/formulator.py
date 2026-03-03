from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.states import AgenOptimicState, OptimizationModel
from app.prompts.prompts import FORMULATOR_SYSTEM_PROMPT
from app.schemas.outputs import FormulatorOutput
from app.utils.llm_factory import get_llm


def run_formulator(state: AgenOptimicState) -> dict:
    llm = get_llm("formulator").with_structured_output(FormulatorOutput)

    messages = [
        SystemMessage(content=FORMULATOR_SYSTEM_PROMPT),
        HumanMessage(content=state.question),
    ]

    result: FormulatorOutput = llm.invoke(messages)

    return {
        "optimization_model": OptimizationModel(
            variables=result.variables,
            constraints=result.constraints,
            objective_function=result.objective_function,
        )
    }
