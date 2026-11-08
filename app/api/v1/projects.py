"""Projects (owner-scoped CRUD on meta_db)."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api.deps import CurrentUser, DbSession
from app.core.exceptions import AppError, ErrorCode
from app.db.models import Project
from app.schemas.user import ProjectIn, ProjectOut

router = APIRouter(prefix="/projects", tags=["Projects"])
ProjectId = Annotated[int, Path(ge=1)]


def _owned(db: DbSession, user_id: int, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None or project.owner_id != user_id:
        raise AppError(ErrorCode.NOT_FOUND, "Project not found.")
    return project


@router.get("", response_model=list[ProjectOut])
def list_projects(db: DbSession, user: CurrentUser) -> list[Project]:
    stmt = select(Project).where(Project.owner_id == user.id).order_by(Project.created_at.desc())
    return list(db.execute(stmt).scalars())


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectIn, db: DbSession, user: CurrentUser) -> Project:
    project = Project(owner_id=user.id, name=body.name.strip(), description=body.description)
    db.add(project)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise AppError(ErrorCode.CONFLICT, "You already have a project with this name.") from None
    db.refresh(project)
    return project


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: ProjectId, db: DbSession, user: CurrentUser) -> Project:
    return _owned(db, user.id, project_id)


@router.put("/{project_id}", response_model=ProjectOut)
def update_project(project_id: ProjectId, body: ProjectIn, db: DbSession,
                   user: CurrentUser) -> Project:
    project = _owned(db, user.id, project_id)
    project.name, project.description = body.name.strip(), body.description
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise AppError(ErrorCode.CONFLICT, "You already have a project with this name.") from None
    db.refresh(project)
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: ProjectId, db: DbSession, user: CurrentUser) -> None:
    db.delete(_owned(db, user.id, project_id))
    db.commit()
