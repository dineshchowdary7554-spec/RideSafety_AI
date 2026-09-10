import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
FL_DATA_DIR = BASE_DIR / "data" / "fl"
RIDES_DIR = FL_DATA_DIR / "rides"
DATABASE_PATH = FL_DATA_DIR / "fl_database.db"
MODEL_PATH = FL_DATA_DIR / "global_model.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    FL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DATABASE_PATH), timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_fl_database():
    FL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RIDES_DIR.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS fl_clients (
                client_id TEXT PRIMARY KEY,
                app_version TEXT NOT NULL,
                platform TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                last_seen TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS fl_rounds (
                round_id INTEGER PRIMARY KEY,
                model_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                required_clients INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS fl_updates (
                update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                model_update TEXT NOT NULL,
                metadata TEXT,
                submitted_at TEXT NOT NULL,
                UNIQUE(client_id, round_id)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS global_models (
                version INTEGER PRIMARY KEY,
                model_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                round_id INTEGER NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS fl_ride_datasets (
                dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id TEXT NOT NULL,
                ride_id TEXT NOT NULL,
                raw_file_path TEXT NOT NULL,
                feature_file_path TEXT NOT NULL,
                input_records INTEGER NOT NULL,
                accepted_records INTEGER NOT NULL,
                rejected_records INTEGER NOT NULL,
                labelled_records INTEGER NOT NULL,
                readiness_score REAL NOT NULL,
                ready_for_training INTEGER NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(client_id, ride_id)
            )
        """)
        connection.commit()
    initialize_default_model()


def initialize_default_model():
    if MODEL_PATH.exists():
        return
    save_global_model(1, {
        "version": 1,
        "parameters": {},
        "status": "pipeline_ready_not_connected_to_production_model",
        "createdAt": utc_now(),
    }, 0)


def register_client(client_id: str, app_version: str, platform: str):
    now = utc_now()
    with get_connection() as connection:
        connection.execute("""
            INSERT INTO fl_clients (client_id, app_version, platform, registered_at, last_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(client_id) DO UPDATE SET
                app_version=excluded.app_version,
                platform=excluded.platform,
                last_seen=excluded.last_seen
        """, (client_id, app_version, platform, now, now))
        connection.commit()


def client_exists(client_id: str) -> bool:
    with get_connection() as connection:
        return connection.execute("SELECT 1 FROM fl_clients WHERE client_id=? LIMIT 1", (client_id,)).fetchone() is not None


def get_registered_client_count() -> int:
    with get_connection() as connection:
        return connection.execute("SELECT COUNT(*) FROM fl_clients").fetchone()[0]


def get_latest_round() -> Optional[Dict[str, Any]]:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM fl_rounds ORDER BY round_id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def get_round(round_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM fl_rounds WHERE round_id=?", (round_id,)).fetchone()
    return dict(row) if row else None


def create_round(round_id: int, model_version: int, required_clients: int):
    with get_connection() as connection:
        connection.execute("""
            INSERT INTO fl_rounds (round_id, model_version, status, required_clients, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (round_id, model_version, "WAITING_FOR_UPDATES", required_clients, utc_now(), None))
        connection.commit()


def get_round_update_count(round_id: int) -> int:
    with get_connection() as connection:
        return connection.execute("SELECT COUNT(*) FROM fl_updates WHERE round_id=?", (round_id,)).fetchone()[0]


def save_model_update(client_id: str, round_id: int, sample_count: int, model_update: Dict[str, float], metadata: Optional[Dict[str, Any]]) -> bool:
    try:
        with get_connection() as connection:
            connection.execute("""
                INSERT INTO fl_updates (client_id, round_id, sample_count, model_update, metadata, submitted_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (client_id, round_id, sample_count, json.dumps(model_update), json.dumps(metadata or {}), utc_now()))
            connection.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_round_updates(round_id: int) -> List[Dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM fl_updates WHERE round_id=? ORDER BY update_id ASC", (round_id,)).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["model_update"] = json.loads(item["model_update"])
        item["metadata"] = json.loads(item["metadata"] or "{}")
        items.append(item)
    return items


def save_global_model(version: int, model_data: Dict[str, Any], round_id: int):
    model_data = dict(model_data)
    model_data["version"] = version
    model_data["updatedAt"] = utc_now()
    FL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model_data, indent=2), encoding="utf-8")
    with get_connection() as connection:
        connection.execute("""
            INSERT OR REPLACE INTO global_models (version, model_data, created_at, round_id)
            VALUES (?, ?, ?, ?)
        """, (version, json.dumps(model_data), utc_now(), round_id))
        connection.commit()


def get_global_model() -> Dict[str, Any]:
    initialize_default_model()
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def complete_round(round_id: int):
    with get_connection() as connection:
        connection.execute("""
            UPDATE fl_rounds SET status=?, completed_at=?
            WHERE round_id=? AND status='WAITING_FOR_UPDATES'
        """, ("COMPLETED", utc_now(), round_id))
        connection.commit()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def save_ride_dataset(client_id: str, ride_id: str, records: List[Dict[str, Any]], feature_row: Dict[str, Any], summary: Dict[str, Any], readiness: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    client_dir = RIDES_DIR / _safe_name(client_id)
    client_dir.mkdir(parents=True, exist_ok=True)
    base = _safe_name(ride_id)
    raw_path = client_dir / f"{base}_raw.json"
    feature_path = client_dir / f"{base}_features.json"
    raw_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    feature_payload = {"rideId": ride_id, "clientId": client_id, "summary": summary, "readiness": readiness, "features": feature_row}
    feature_path.write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")
    with get_connection() as connection:
        connection.execute("""
            INSERT OR REPLACE INTO fl_ride_datasets (
                client_id, ride_id, raw_file_path, feature_file_path,
                input_records, accepted_records, rejected_records, labelled_records,
                readiness_score, ready_for_training, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            client_id, ride_id, str(raw_path), str(feature_path),
            int(summary["inputRecords"]), int(summary["acceptedRecords"]), int(summary["rejectedRecords"]), int(summary["labelledRecords"]),
            float(readiness["score"]), 1 if readiness["ready"] else 0,
            json.dumps(metadata or {}), utc_now()
        ))
        connection.commit()
    return {"rawFile": str(raw_path), "featureFile": str(feature_path)}


def get_client_datasets(client_id: str) -> List[Dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM fl_ride_datasets WHERE client_id=? ORDER BY dataset_id DESC", (client_id,)).fetchall()
    return [dict(row) for row in rows]


def get_all_dataset_stats() -> Dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute("""
            SELECT COUNT(*) AS rides,
                   COALESCE(SUM(accepted_records), 0) AS records,
                   COALESCE(SUM(labelled_records), 0) AS labelled,
                   COALESCE(AVG(readiness_score), 0) AS average_readiness,
                   COALESCE(SUM(ready_for_training), 0) AS training_ready_rides
            FROM fl_ride_datasets
        """).fetchone()
    return dict(row)


def load_client_feature_rows(client_id: str) -> List[Dict[str, Any]]:
    rows = get_client_datasets(client_id)
    output = []
    for row in rows:
        path = Path(row["feature_file_path"])
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            output.append(payload["features"])
    return output
