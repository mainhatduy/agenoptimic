from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph.graph import compiled_graph
from app.graph.states import AgenOptimicState, OptimizationModel


class SolveRequest(BaseModel):
    question: str


class OptimizationModelResponse(BaseModel):
    variables: str
    constraints: str
    objective_function: str


class SolveResponse(BaseModel):
    question: str
    optimization_model: Optional[OptimizationModelResponse] = None
    plan: Optional[str] = None
    generated_code: Optional[str] = None
    solution: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load config on startup to surface misconfigurations early.
    from app.utils.llm_factory import get_settings
    get_settings()
    yield


app = FastAPI(
    title="AgenOptimic",
    description="Multi-agent system for solving combinatorial optimization problems.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest) -> SolveResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    initial_state = AgenOptimicState(question=request.question)

    try:
        result: AgenOptimicState = await compiled_graph.ainvoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {exc}") from exc

    opt_model_resp = None
    if result.optimization_model:
        opt_model_resp = OptimizationModelResponse(
            variables=result.optimization_model.variables,
            constraints=result.optimization_model.constraints,
            objective_function=result.optimization_model.objective_function,
        )

    return SolveResponse(
        question=result.question,
        optimization_model=opt_model_resp,
        plan=result.plan,
        generated_code=result.generated_code,
        solution=result.solution,
        error=result.error,
        retry_count=result.retry_count,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
