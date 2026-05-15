"""ZEV-UP Scotland segment simulator router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..agents.zevup_compare import compare_scotland_segments
from ..schemas.zevup import ZEVUPCompareRequest, ZEVUPCompareResponse

router = APIRouter(prefix="/api/zevup", tags=["zevup"])


@router.post("/scotland/segments/compare", response_model=ZEVUPCompareResponse)
def compare_scotland(req: ZEVUPCompareRequest) -> ZEVUPCompareResponse:
    try:
        return compare_scotland_segments(req)
    except KeyError as exc:
        key = exc.args[0] if exc.args else "unknown resource"
        raise HTTPException(
            status_code=404,
            detail=f"ZEV-UP resource not found: {key}",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Required ZEV-UP source data is missing.",
        ) from exc
    except ValueError as exc:
        message = str(exc)
        if "Unknown intervention" in message:
            raise HTTPException(status_code=400, detail=message) from exc
        raise HTTPException(
            status_code=500,
            detail="ZEV-UP comparison failed.",
        ) from exc
