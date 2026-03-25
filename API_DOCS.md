# Pose Scoring API — Documentation

> Base URL: `https://backend-fastapi-kaya-production.up.railway.app`
> Version: 1.0.0

---

## Overview

REST API for scoring exercise pose quality using LSTM and CNN autoencoder models.
Accepts raw MediaPipe landmark frames and returns quality scores (0–100).

Available poses: `ArmRaise`, `KneeRaise`, `Push-up`, `SquatArmRaise`, `TorsoTwist`

---

## Endpoints

### GET /health

Health check. Returns service status and available poses.

**Response 200**
```json
{
  "ok": true,
  "available_poses": ["ArmRaise", "KneeRaise", "Push-up", "SquatArmRaise", "TorsoTwist"]
}
```

---

### GET /poses

List available pose model names.

**Response 200**
```json
{
  "success": true,
  "poses": ["ArmRaise", "KneeRaise", "Push-up", "SquatArmRaise", "TorsoTwist"]
}
```

---

### POST /predict

Score a pose session. Send landmark frames, get quality scores back.

**Request Headers**
```
Content-Type: application/json
```

**Request Body**
```json
{
  "pose_name": "KneeRaise",
  "raw_frames": [
    [[0.51, 0.23], [0.48, 0.23], [0.50, 0.40], ...],
    [[0.52, 0.24], [0.47, 0.24], [0.51, 0.41], ...]
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `pose_name` | `string` | yes | One of the available poses (case-insensitive) |
| `raw_frames` | `float[][][]` | yes | Shape **(N, 33, 2)** — N frames, 33 MediaPipe landmarks, `[x, y]` per landmark |

> **Constraints:**
> - Minimum **2 frames** required
> - Each frame must have exactly **33 landmarks**
> - Each landmark must have exactly **2 values** (x, y) — normalized 0.0–1.0
> - No `NaN` or `Infinity` values
> - No extra fields allowed in the body

**Response 200 — Success**
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

| Field | Type | Description |
|---|---|---|
| `lstm_score` | `float` | LSTM autoencoder score (0–100) |
| `cnn_score` | `float` | CNN autoencoder score (0–100) |
| `avg_score` | `float` | Weighted average: `lstm×0.3 + cnn×0.7` (0–100) |
| `lstm_error` | `float` | Raw LSTM reconstruction error (MSE) |
| `cnn_error` | `float` | Raw CNN reconstruction error (MSE) |
| `move_gate` | `float` | Movement gate multiplier (0.0–1.0) |
| `move_ratio` | `float` | Raw movement ratio before clipping |

---

## Error Response Format

All errors use this envelope:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message.",
    "details": {}
  }
}
```

### Error Codes

| HTTP | Code | Cause |
|---|---|---|
| 404 | `UNKNOWN_POSE` | `pose_name` not found in loaded models |
| 422 | `REQUEST_VALIDATION_ERROR` | Missing/wrong-type fields |
| 422 | `INVALID_INPUT_FORMAT` | `raw_frames` is not numeric |
| 422 | `INVALID_INPUT_SHAPE` | Array shape is not `(N, 33, 2)` |
| 422 | `INSUFFICIENT_FRAMES` | Fewer than 2 frames provided |
| 422 | `NON_FINITE_VALUES` | `raw_frames` contains `NaN` or `Infinity` |
| 500 | `MODEL_INFERENCE_ERROR` | TFLite inference failed |
| 500 | `INTERNAL_SERVER_ERROR` | Unhandled exception |

---

## TypeScript Types

```typescript
type PoseName = "ArmRaise" | "KneeRaise" | "Push-up" | "SquatArmRaise" | "TorsoTwist";

// Single landmark: [x, y] normalized 0.0–1.0
type Landmark = [number, number];

// Single frame: 33 MediaPipe landmarks
type Frame = Landmark[]; // length = 33

interface PredictRequest {
  pose_name: PoseName;
  raw_frames: Frame[]; // minimum 2 frames
}

interface ScorePayload {
  lstm_score: number;   // 0–100
  cnn_score: number;    // 0–100
  avg_score: number;    // 0–100, weighted (lstm×0.3 + cnn×0.7)
  lstm_error: number;   // raw MSE
  cnn_error: number;    // raw MSE
  move_gate: number;    // 0.0–1.0
  move_ratio: number;
}

interface PredictSuccessResponse {
  success: true;
  pose_name: string;
  scores: ScorePayload;
}

interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

interface PredictErrorResponse {
  success: false;
  error: ApiError;
}

type PredictResponse = PredictSuccessResponse | PredictErrorResponse;

interface HealthResponse {
  ok: boolean;
  available_poses: PoseName[];
}

interface PosesResponse {
  success: boolean;
  poses: PoseName[];
}
```

---

## Usage Example (TypeScript / fetch)

```typescript
const BASE_URL = "https://backend-fastapi-kaya-production.up.railway.app";

async function scorepose(
  poseName: PoseName,
  frames: Frame[]
): Promise<PredictResponse> {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      pose_name: poseName,
      raw_frames: frames,
    }),
  });

  const data: PredictResponse = await res.json();
  return data;
}

// Example usage
const frames: Frame[] = mediaposeResults.map((result) =>
  result.poseLandmarks.map((lm) => [lm.x, lm.y])
);

const result = await scorepose("KneeRaise", frames);

if (result.success) {
  console.log("Score:", result.scores.avg_score);
} else {
  console.error("Error:", result.error.code, result.error.message);
}
```

---

## Frame Resampling (Frontend)

### Why resample?

The backend resamples all input to **30 frames** internally (via linear interpolation).
However, sending inconsistent raw frame counts — e.g., sometimes 10 frames and sometimes 300 frames for the same motion — can cause results to drift even for identical movements, because the temporal density of motion captured differs.

**Best practice:** always send frames representing **exactly one complete repetition**, resampled to the per-pose target count below.

### Recommended raw frames per pose (one rep at ~30 fps)

| Pose | Target frames |
|---|---|
| `ArmRaise` | 45 |
| `KneeRaise` | 70 |
| `Push-up` | 22 |
| `SquatArmRaise` | 45 |
| `TorsoTwist` | 29 |

### Resampling utility (TypeScript)

```typescript
type PoseName = "ArmRaise" | "KneeRaise" | "Push-up" | "SquatArmRaise" | "TorsoTwist";
type Landmark = [number, number];  // [x, y] normalized 0.0–1.0
type Frame = Landmark[];           // 33 landmarks per frame

// Target frame count per pose — represents one complete repetition at ~30 fps.
// Always resample collected frames to this count before calling /predict,
// so the backend receives consistent temporal density regardless of recording speed.
const POSE_TARGET_FRAMES: Record<PoseName, number> = {
  ArmRaise: 45,
  KneeRaise: 70,
  "Push-up": 22,
  SquatArmRaise: 45,
  TorsoTwist: 29,
};

/**
 * Resample `frames` to exactly `targetCount` frames using linear interpolation.
 * Each landmark coordinate is interpolated independently across time.
 *
 * @param frames     - Collected frames from MediaPipe, shape (N, 33, 2)
 * @param targetCount - Desired output frame count
 * @returns Resampled frames, shape (targetCount, 33, 2)
 */
function resampleFrames(frames: Frame[], targetCount: number): Frame[] {
  const N = frames.length;

  if (N === 0) throw new Error("No frames to resample.");
  if (N === targetCount) return frames;
  if (N === 1) return Array(targetCount).fill(frames[0]);

  const numLandmarks = frames[0].length; // 33

  return Array.from({ length: targetCount }, (_, outIdx) => {
    // Map output index to a continuous position in the source array
    const srcPos = (outIdx / (targetCount - 1)) * (N - 1);
    const lo = Math.floor(srcPos);
    const hi = Math.min(lo + 1, N - 1);
    const t = srcPos - lo; // interpolation weight [0, 1)

    return Array.from({ length: numLandmarks }, (_, lmIdx) => {
      const [x0, y0] = frames[lo][lmIdx];
      const [x1, y1] = frames[hi][lmIdx];
      return [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t] as Landmark;
    }) as Frame;
  });
}

/**
 * Validate that all frames have shape (N, 33, 2) and no NaN/Infinity values.
 * Throws a descriptive error if validation fails.
 */
function validateFrames(frames: Frame[]): void {
  if (frames.length < 2) throw new Error("Need at least 2 frames.");

  for (let i = 0; i < frames.length; i++) {
    if (frames[i].length !== 33) {
      throw new Error(`Frame ${i} has ${frames[i].length} landmarks, expected 33.`);
    }
    for (let j = 0; j < 33; j++) {
      const [x, y] = frames[i][j];
      if (!isFinite(x) || !isFinite(y)) {
        throw new Error(`Frame ${i}, landmark ${j} contains non-finite value.`);
      }
    }
  }
}

/**
 * Prepare frames for /predict:
 * 1. Validate shape (N, 33, 2)
 * 2. Resample to the pose-specific target frame count
 *
 * @param frames   - Raw collected frames from MediaPipe
 * @param poseName - The exercise pose name
 * @returns Resampled frames ready to send as `raw_frames`
 */
function prepareFrames(frames: Frame[], poseName: PoseName): Frame[] {
  validateFrames(frames);
  const target = POSE_TARGET_FRAMES[poseName];
  return resampleFrames(frames, target);
}
```

### Full usage example

```typescript
const BASE_URL = "https://backend-fastapi-kaya-production.up.railway.app";

async function scorePose(
  poseName: PoseName,
  rawFrames: Frame[]
): Promise<PredictResponse> {
  // Resample to consistent frame count before sending
  const frames = prepareFrames(rawFrames, poseName);

  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pose_name: poseName, raw_frames: frames }),
  });

  return res.json() as Promise<PredictResponse>;
}

// --- In your MediaPipe callback ---
const collectedFrames: Frame[] = [];

// While recording, push each frame:
function onPoseResult(result: PoseLandmarkerResult) {
  const frame = result.landmarks[0].map((lm) => [lm.x, lm.y] as Landmark);
  collectedFrames.push(frame);
}

// When one rep is complete, score it:
async function onRepComplete() {
  const result = await scorePose("KneeRaise", collectedFrames);
  collectedFrames.length = 0; // reset for next rep

  if (result.success) {
    console.log("Score:", result.scores.avg_score);
    // move_gate near 0 = user barely moved, scores unreliable
    if (result.scores.move_gate < 0.3) {
      console.warn("Low movement detected — ask user to move more.");
    }
  } else {
    console.error(result.error.code, result.error.message);
  }
}
```

---

## Notes for Frontend Integration

- **Landmark source:** Use MediaPipe Pose (`@mediapipe/pose` or `@mediapipe/tasks-vision`). Each frame should be the `.poseLandmarks` array — 33 points with `x` and `y` (normalized 0–1).
- **Frame collection:** Collect frames for **one complete repetition**, then call `prepareFrames()` before sending to `/predict`.
- **Score interpretation:**
  - `avg_score >= 80` — Good form
  - `avg_score 50–79` — Needs improvement
  - `avg_score < 50` — Poor form
- **move_gate:** Close to `0.0` means the user barely moved; scores will be near zero regardless of form quality.
- **CORS:** Fully open — no special headers needed from the browser.
