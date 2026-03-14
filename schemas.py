from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    pose_name: str = Field(..., description="Pose name from frontend, e.g. ArmRaise")
    raw_frames: List[List[List[float]]] = Field(
        ..., description="Raw landmarks with shape (N, 33, 2)."
    )

    class Config:
        extra = "forbid"


class ScorePayload(BaseModel):
    lstm_score: float
    cnn_score: float
    avg_score: float
    lstm_error: float
    cnn_error: float
    move_gate: float
    move_ratio: float


class PredictSuccessResponse(BaseModel):
    success: bool = True
    pose_name: str
    scores: ScorePayload


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
