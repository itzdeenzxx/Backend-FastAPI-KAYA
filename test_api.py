import json
import csv
import urllib.request
import urllib.error

CSV_FILE = "./data/ArmRaise/ArmRaise_20260227_212943.csv"
URL = "https://backend-fastapi-kaya-production.up.railway.app/predict"

frames = []
with open(CSV_FILE, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame_landmarks = []
        for i in range(33):
            x = float(row[f"x{i}"])
            y = float(row[f"y{i}"])
            frame_landmarks.append([x, y])
        frames.append(frame_landmarks)

payload = {
    "pose_name": "ArmRaise",
    "raw_frames": frames
}

json_data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(URL, data=json_data, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        print("Status:", response.status)
        print("Response:", response.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    print(f"HTTPError: {e.code} - {e.reason}")
    print("Error Response body:", e.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
