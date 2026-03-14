import csv
import json
import sys

import requests


def csv_to_frames(csv_path: str):
    frames = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            values = [float(v) for v in row[2:]]
            landmarks = [[values[i], values[i + 1]] for i in range(0, 66, 2)]
            frames.append(landmarks)
    return frames


def main():
    if len(sys.argv) < 3:
        print("ใช้: python test_with_csv.py <csv_file> <pose_name> [api_url]")
        print("ตัวอย่าง: python test_with_csv.py ../data/Push-up/Push-up_20260305_211806.csv Push-up")
        sys.exit(1)

    csv_path = sys.argv[1]
    pose_name = sys.argv[2]
    api_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"

    # แปลง CSV → frames
    frames = csv_to_frames(csv_path)
    print(f"📂 อ่านได้ {len(frames)} เฟรม จาก {csv_path}")
    print(f"🏋️ ท่า: {pose_name}")
    print(f"🌐 API: {api_url}")
    print()

    # ยิง /health ก่อน
    print("--- Health Check ---")
    resp = requests.get(f"{api_url}/health")
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    print()

    # ยิง /predict
    print("--- Predict ---")
    payload = {"pose_name": pose_name, "raw_frames": frames}
    resp = requests.post(f"{api_url}/predict", json=payload)
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
