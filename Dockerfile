FROM python:3.12-slim

# Model cache lives inside the image so the first request doesn't stall on a
# ~90MB download. HF_HOME must be set before sentence-transformers is imported.
ENV HF_HOME=/app/.hf_cache \
    PYTHONUNBUFFERED=1 \
    CHROMA_DIR=/app/chroma_db \
    API_URL=http://localhost:8000

WORKDIR /app

# CPU-only torch: the default wheel pulls ~2GB of CUDA libraries that are dead
# weight in a CPU container.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY rag_core.py api.py streamlit_app.py RAG_Agent.py ./

# Background art. The UI falls back to a gradient when this is empty.
COPY assets/ ./assets/

# Pre-built vector store — 3,947 chunks already embedded. Baking it in means
# the container never re-embeds on boot and the source texts needn't ship.
COPY chroma_db/ ./chroma_db/

# Warm the embedding model into the image layer.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY start.sh .
RUN chmod +x start.sh

# 7860 is the port HuggingFace Spaces expects; 8000 is the API.
EXPOSE 7860 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5).raise_for_status()"

CMD ["./start.sh"]
