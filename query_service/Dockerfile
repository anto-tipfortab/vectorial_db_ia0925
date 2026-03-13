FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py streamlit_app.py ./

# Launch FastAPI (8000) and Streamlit (8501) in parallel
CMD uvicorn main:app --host 0.0.0.0 --port 8000 & \
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true & \
    wait