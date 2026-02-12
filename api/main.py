from fastapi import FastAPI
from sqlalchemy import text
from .database import engine

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/test-db")
def test_db():
    with engine.connect() as conn:
        val = conn.execute(text("SELECT 1")).scalar_one()
        return {"db": val}
