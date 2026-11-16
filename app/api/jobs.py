"""作业 API:前端只传作业名,命令一律走 app/core/jobs.py 的白名单注册表。"""
from fastapi import APIRouter, HTTPException

from app.core import jobs

router = APIRouter(prefix="/api/jobs")


@router.get("")
async def list_jobs() -> dict:
    return {"jobs": jobs.list_jobs()}


@router.post("/{name}")
async def start_job(name: str) -> dict:
    try:
        return jobs.start(name)
    except jobs.UnknownJob:
        raise HTTPException(status_code=404, detail=f"未知作业:{name}")
    except jobs.JobRunning:
        raise HTTPException(status_code=409, detail=f"作业 {name} 仍在运行,拒绝重入")


@router.get("/{name}")
async def job_status(name: str) -> dict:
    try:
        st = jobs.status(name)
    except jobs.UnknownJob:
        raise HTTPException(status_code=404, detail=f"未知作业:{name}")
    return {**st, "log_tail": jobs.tail(name)}


@router.post("/{name}/stop")
async def stop_job(name: str) -> dict:
    try:
        return jobs.stop(name)
    except jobs.UnknownJob:
        raise HTTPException(status_code=404, detail=f"未知作业:{name}")
    except jobs.JobNotRunning:
        raise HTTPException(status_code=409, detail=f"作业 {name} 未在运行")
