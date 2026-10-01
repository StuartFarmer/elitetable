"""Organism evidence and generation leaderboards saved through Run.save()."""

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Organism(SQLModel, table=True):
    __tablename__ = "elitesearch_organism"
    id: int = Field(primary_key=True)
    generation: int
    kind: str
    parent_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    status: str = "planned"
    policy_id: str | None = None
    name: str = ""
    description: str = ""
    implementation: str = ""
    score: float | None = None
    seed_scores: dict[str, float] = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None
    repairs: int = 0
    calls: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    revisions: list[dict] = Field(default_factory=list, sa_column=Column(JSON))


class Generation(SQLModel, table=True):
    __tablename__ = "elitesearch_generation"
    number: int = Field(primary_key=True)
    status: str = "running"
    elite_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    promoted_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    error: str | None = None
