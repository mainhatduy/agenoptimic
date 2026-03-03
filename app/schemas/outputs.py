from pydantic import BaseModel, Field


class FormulatorOutput(BaseModel):
    variables: str = Field(
        description="Decision variables of the optimization problem (names, types, domains)."
    )
    constraints: str = Field(
        description="Mathematical constraints of the optimization problem."
    )
    objective_function: str = Field(
        description="The objective function to minimize or maximize."
    )


class PlannerOutput(BaseModel):
    plan: str = Field(
        description=(
            "Step-by-step plan describing which Python library to use "
            "(e.g. PuLP, SciPy, CVXPY) and how to model and solve the problem."
        )
    )


class CoderOutput(BaseModel):
    code: str = Field(
        description=(
            "Complete, self-contained Python script that solves the optimization problem "
            "and prints the final answer to stdout."
        )
    )
