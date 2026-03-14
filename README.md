# FastAPI Pose Scoring Service

This service receives pose name and raw landmarks from frontend, then predicts pose quality scores (`%`) using exercise-specific TFLite autoencoders.

## Features

- Supports all available pose folders under `models/`.
- Frontend selects which model to use by sending `pose_name`.
- Uses scoring logic similar to `pose_scoring_app.py`.
- Returns structured error payload when request/model/config has issues.

## Install

```bash
pip install fastapi uvicorn numpy scipy pydantic
```

If you do not have `tflite-runtime`, install TensorFlow instead:

```bash
pip install tensorflow
```

## Run

From workspace root:

```bash
uvicorn fastapi_service.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- `GET /health` - service status + available poses
- `GET /poses` - list available pose names
- `POST /predict` - score one request

## Request Example

```json
{
  "pose_name": "KneeRaise",
  "raw_frames": [
    [[0.51, 0.23], [0.48, 0.23], [0.50, 0.25]],
    [[0.52, 0.24], [0.47, 0.24], [0.50, 0.26]]
  ]
}
```

`raw_frames` must be shape `(N, 33, 2)`.

## Success Response Example

```json
{
  "success": true,
  "pose_name": "KneeRaise",
  "scores": {
    "lstm_score": 81.7,
    "cnn_score": 89.4,
    "avg_score": 87.1,
    "lstm_error": 0.003211,
    "cnn_error": 0.002991,
    "move_gate": 0.942,
    "move_ratio": 0.2824
  }
}
```

## Error Response Example

```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT_SHAPE",
    "message": "raw_frames must have shape (N, 33, 2).",
    "details": {
      "received_shape": [30, 22, 2]
    }
  }
}
```

## Frontend Notes

- Send exact pose name from `/poses` if possible.
- Common aliases are accepted (case-insensitive and dash-insensitive).
- Always check `success` field and handle error payload on failure.
