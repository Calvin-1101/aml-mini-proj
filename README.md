frontend
http://127.0.0.1:8000/

uv run uvicorn app.main:app --reload --port 8000


mlflow
http://127.0.0.1:5000/

to open mlflow ui: 
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000