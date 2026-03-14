FROM python:3.11-slim

WORKDIR /app

# คัดลอกเฉพาะไฟล์ requirements
COPY requirements.txt ./requirements.txt

# อัปเดตและติดตั้งไลบรารี
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir tensorflow-cpu

# คัดลอกไฟล์ Python source
COPY main.py service.py schemas.py ./

# คัดลอกโฟลเดอร์ Models
COPY models/ ./models/

# ตั้งค่าพอร์ต
EXPOSE 8000

# รัน FastAPI (ใช้ $PORT จาก Railway, fallback เป็น 8000 สำหรับ local dev)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
