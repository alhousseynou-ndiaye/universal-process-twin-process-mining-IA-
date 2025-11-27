import os
import json
import sqlite3
import hashlib
import secrets
from datetime import datetime
from typing import Dict, Any

import pandas as pd
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    APIRouter,
    Form,
    Header,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

from process_analyzer import analyze_dataframe
from ai_report import generate_ai_report, generate_automation_ideas

# ================== CONFIG GROQ ==================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY est manquant. Ajoute-le dans un fichier .env.")

client = Groq(api_key=GROQ_API_KEY)

# ================== DB AUTH (SQLite) ==================
DB_PATH = "upt_auth.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


def get_current_user(x_auth_token: str = Header(None)):
    if not x_auth_token:
        raise HTTPException(
            status_code=401,
            detail="Token d'authentification manquant. Connecte-toi d'abord.",
        )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.id, u.email
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.token = ?
        """,
        (x_auth_token,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(
            status_code=401,
            detail="Session invalide ou expirée. Reconnecte-toi.",
        )

    return {"id": row[0], "email": row[1]}


init_db()

router = APIRouter()

app = FastAPI(
    title="Universal Process Twin API",
    version="0.4.0",
    description="API d'analyse de processus à partir de fichiers CSV/Excel, avec IA, KPI et idées d'automatisation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Bienvenue sur l'API Universal Process Twin v0.4",
        "auth": {
            "POST /register": "Création de compte",
            "POST /login": "Connexion et obtention d'un token",
        },
        "process": {
            "POST /analyze": "Analyse du processus (stats + graph + kpi) [auth requis]",
            "POST /analyze_with_ai": "Analyse + rapport IA [auth requis]",
            "POST /analyze_mapped": "Analyse avec mapping de colonnes [auth requis]",
            "POST /detect_columns": "Détection des colonnes [auth requis]",
            "POST /guess_structure": "Détection IA structure [auth requis]",
            "POST /suggest_automations": "Idées d'automatisation IA à partir de l'analyse [auth requis]",
        },
    }


# ================== AUTH ==================
@app.post("/register")
async def register(email: str = Form(...), password: str = Form(...)):
    if len(password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Le mot de passe doit contenir au moins 6 caractères.",
        )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, hash_password(password), datetime.utcnow().isoformat()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="Un compte existe déjà avec cet email.",
        )
    conn.close()
    return {"message": "Compte créé avec succès.", "email": email}


@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()

    if not row or not verify_password(password, row[1]):
        raise HTTPException(
            status_code=401,
            detail="Email ou mot de passe incorrect.",
        )

    user_id = row[0]
    token = secrets.token_hex(32)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user_id, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()

    return {"token": token, "email": email}


# ================== ANALYSE ==================
@app.post("/analyze")
async def analyze_process(
    file: UploadFile = File(...), user=Depends(get_current_user)
):
    filename = file.filename or ""
    if not filename.lower().endswith((".csv", ".xls", ".xlsx")):
        raise HTTPException(
            status_code=415,
            detail="Format non supporté. Merci d'envoyer un fichier .csv, .xls ou .xlsx.",
        )

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Impossible de lire le fichier : {e}"
        )

    required_cols = {"case_id", "step"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail=f"Le fichier doit contenir au minimum les colonnes : {required_cols}.",
        )

    result = analyze_dataframe(df)
    return result


@app.post("/analyze_with_ai")
async def analyze_process_with_ai(
    file: UploadFile = File(...), user=Depends(get_current_user)
):
    filename = file.filename or ""
    if not filename.lower().endswith((".csv", ".xls", ".xlsx")):
        raise HTTPException(
            status_code=415,
            detail="Format non supporté. Merci d'envoyer un fichier .csv, .xls ou .xlsx.",
        )

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Impossible de lire le fichier : {e}"
        )

    required_cols = {"case_id", "step"}
    if not required_cols.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail=f"Le fichier doit contenir au minimum les colonnes : {required_cols}.",
        )

    result = analyze_dataframe(df)

    try:
        report = generate_ai_report(result["stats"], result["graph"])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération du rapport IA : {e}",
        )

    result["report"] = report
    return result


@app.post("/analyze_mapped")
async def analyze_mapped(
    file: UploadFile = File(...),
    case_col: str = Form(...),
    step_col: str = Form(...),
    ts_col: str = Form(""),
    use_ai: bool = Form(False),
    user=Depends(get_current_user),
):
    filename = file.filename or ""
    if not filename.lower().endswith((".csv", ".xls", ".xlsx")):
        raise HTTPException(
            status_code=415,
            detail="Format non supporté. Merci d'envoyer un fichier .csv, .xls ou .xlsx.",
        )

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Impossible de lire le fichier : {e}"
        )

    missing = []
    for colname, label in [(case_col, "case_col"), (step_col, "step_col")]:
        if colname not in df.columns:
            missing.append(f"{label}='{colname}'")
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Les colonnes suivantes n'existent pas dans le fichier : {', '.join(missing)}",
        )

    df2 = df.copy()
    rename_map = {
        case_col: "case_id",
        step_col: "step",
    }
    if ts_col and ts_col in df2.columns:
        rename_map[ts_col] = "timestamp"

    df2 = df2.rename(columns=rename_map)

    if "timestamp" in df2.columns:
        df2["timestamp"] = pd.to_datetime(df2["timestamp"], errors="coerce")

    result = analyze_dataframe(df2)

    if use_ai:
        try:
            report = generate_ai_report(result["stats"], result["graph"])
            result["report"] = report
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de la génération du rapport IA : {e}",
            )

    return result


# ================== DETECTION COLONNES + IA STRUCTURE ==================
@router.post("/detect_columns")
async def detect_columns(
    file: UploadFile = File(...), user=Depends(get_current_user)
):
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(".xls") or file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(
                status_code=400,
                detail="Format non supporté (CSV ou Excel uniquement).",
            )

        cols = list(df.columns)

        if len(cols) == 0:
            raise HTTPException(status_code=400, detail="Aucune colonne détectée.")

        return {"columns": cols}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la lecture du fichier : {str(e)}",
        )


@router.post("/guess_structure")
async def guess_structure(columns: dict, user=Depends(get_current_user)):
    try:
        cols = columns.get("columns", [])
        if not cols:
            raise HTTPException(status_code=400, detail="Aucune colonne reçue.")

        prompt = f"""
        Voici une liste de noms de colonnes provenant d'un fichier Excel ou CSV :
        {cols}

        Ton rôle :
        - Identifier laquelle représente l'identifiant du cas (case_id)
        - Identifier celle qui représente l'étape du processus (step)
        - Identifier celle qui représente un timestamp (timestamp) si elle existe.

        RÈGLES IMPORTANTES :
        - Si tu n'es pas sûr pour un champ, mets null.
        - Réponds UNIQUEMENT avec un JSON VALIDE.
        - PAS de texte avant, PAS de texte après.
        - PAS de commentaire, PAS de phrase, JUSTE le JSON.

        Format JSON attendu :
        {{
          "case_id": "nom_colonne_ou_null",
          "step": "nom_colonne_ou_null",
          "timestamp": "nom_colonne_ou_null",
          "confidence": "score entre 0 et 1"
        }}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un expert en data engineering et process mining. Tu dois répondre en JSON strict.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        ai_output = response.choices[0].message.content or ""

        import re

        def try_parse_json(text: str):
            try:
                return json.loads(text)
            except Exception:
                pass

            try:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            except Exception:
                pass

            return None

        result = try_parse_json(ai_output)

        if result is None:
            lower_cols = [c.lower() for c in cols]

            def find_col(keywords):
                for col, low in zip(cols, lower_cols):
                    if any(k in low for k in keywords):
                        return col
                return None

            case_id_col = find_col(["case", "id", "client", "user", "customer"])
            step_col = find_col(["step", "étape", "stage", "status", "event", "action"])
            ts_col = find_col(["time", "date", "timestamp", "created", "updated"])

            result = {
                "case_id": case_id_col,
                "step": step_col,
                "timestamp": ts_col,
                "confidence": "0.5",
            }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur IA : {str(e)}")


# ================== AUTOMATION IA ==================
class AutomationRequest(BaseModel):
    stats: Dict[str, Any]
    graph: Dict[str, Any]
    kpi: Dict[str, Any] = {}
    domain: str = "generic"


@app.post("/suggest_automations")
async def suggest_automations(
    req: AutomationRequest, user=Depends(get_current_user)
):
    """
    Prend les stats + graph + kpi d'un processus et un contexte métier,
    renvoie des idées d'automatisation générées par l'IA.
    """
    try:
        ideas = generate_automation_ideas(req.stats, req.graph, req.kpi, req.domain)
        return {"ideas": ideas}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur IA (automatisations) : {e}",
        )


app.include_router(router)
