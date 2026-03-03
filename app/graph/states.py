from typing import Optional
from pydantic import BaseModel, Field


class OptimizationModel(BaseModel):
    variables: str
    constraints: str
    objective_function: str


class AgenOptimicState(BaseModel):
    question: str

    # Formulator output
    optimization_model: Optional[OptimizationModel] = None

    # Planner output
    plan: Optional[str] = None

    # Coder output
    generated_code: Optional[str] = None

    # Executor output
    solution: Optional[str] = None
    error: Optional[str] = None

    # Retry tracking
    retry_count: int = Field(default=0)
