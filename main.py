import os
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import ErrorResponse, PredictRequest, PredictSuccessResponse
from service import PoseScoringService, PoseServiceError


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

service = PoseScoringService(MODELS_DIR)
app = FastAPI(title="Pose Scoring API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(PoseServiceError)
def handle_pose_service_error(_: Request, exc: PoseServiceError):
    payload = ErrorResponse(
        success=False,
        error={"code": exc.code, "message": exc.message, "details": exc.details},
    )
    return JSONResponse(status_code=exc.status_code, content=payload.dict())


@app.exception_handler(RequestValidationError)
def handle_validation_error(_: Request, exc: RequestValidationError):
    payload = ErrorResponse(
        success=False,
        error={
            "code": "REQUEST_VALIDATION_ERROR",
            "message": "Request body validation failed.",
            "details": {"errors": exc.errors()},
        },
    )
    return JSONResponse(status_code=422, content=payload.dict())


@app.exception_handler(Exception)
def handle_unexpected_error(_: Request, exc: Exception):
    payload = ErrorResponse(
        success=False,
        error={
            "code": "INTERNAL_SERVER_ERROR",
            "message": "Unexpected server error.",
            "details": {"error": str(exc)},
        },
    )
    return JSONResponse(status_code=500, content=payload.dict())


@app.get("/health")
def health() -> Dict[str, object]:
    return {"ok": True, "available_poses": service.list_poses()}


@app.get("/poses")
def poses() -> Dict[str, object]:
    return {"success": True, "poses": service.list_poses()}


@app.post("/predict", response_model=PredictSuccessResponse)
def predict(payload: PredictRequest):
    scores = service.score(payload.pose_name, payload.raw_frames)
    resolved_pose = service.resolve_pose_name(payload.pose_name)
    return PredictSuccessResponse(success=True, pose_name=resolved_pose, scores=scores)
