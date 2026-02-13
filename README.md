# Running gary Locally

## 1 Clone the repository

```bash
git clone <your-repo-url>
cd gary
```

---

## 2 Create and activate virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4 Create `.env` file

Create a file named `.env` in the project root with the following content:

```
DATABASE_URL=postgresql://gary:gary_pw@localhost:5432/gary_db
```

---

## 5 Start PostgreSQL (Docker)

```bash
docker compose up -d
```

---

## 6 Initialize database (first time only)

### Windows (PowerShell)

```powershell
Get-Content sql/001_init.sql | docker exec -i gary-postgres psql -U gary -d gary_db
```

### Linux / macOS

```bash
docker exec -i gary-postgres psql -U gary -d gary_db < sql/001_init.sql
```

---

## 7 Start FastAPI server

```bash
uvicorn api.main:app --reload
```

---

## 8 Open API documentation

Visit:

```
http://127.0.0.1:8000/docs
```

---

# Reset Database (Fresh Start)

```bash
docker compose down -v
docker compose up -d
```

Then rerun Step 6 (database initialization).

---

# System Flow Overview

```
Program
   ↓
Session
   ↓
ExceptionEvent
   ↓
Adjustment Engine
   ↓
Updated Program + Audit Log
```
