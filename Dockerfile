FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download artifacts from HuggingFace Hub at build time
# (avoids cold-start penalty; update HF_REPO_ID with your repo)
RUN python scripts/download_artifacts.py

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
