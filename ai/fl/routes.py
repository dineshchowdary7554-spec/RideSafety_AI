from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .data_pipeline import build_ride_feature_row, dataset_readiness, validate_records
from .manager import ensure_active_round, get_fl_status, try_aggregate_round
from .models import DatasetExportRequest, FLClientRegistration, FLModelUpdate, FLRoundRequest, RideDatasetUpload
from .storage import (
    client_exists, get_all_dataset_stats, get_client_datasets, get_global_model,
    load_client_feature_rows, register_client, save_model_update, save_ride_dataset
)

router = APIRouter(prefix="/fl", tags=["Federated Learning"])


@router.post("/register")
def register_fl_client(data: FLClientRegistration):
    register_client(data.clientId, data.appVersion, data.platform)
    active_round = ensure_active_round()
    return {"success": True, "clientId": data.clientId, "currentRound": active_round["round_id"], "message": "FL client registered successfully."}


@router.get("/status")
def fl_status():
    return get_fl_status()


@router.get("/model")
def get_fl_model():
    return {"success": True, "model": get_global_model()}


@router.get("/round")
def get_active_round():
    active = ensure_active_round()
    return {"success": True, "roundId": active["round_id"], "modelVersion": active["model_version"], "status": active["status"], "requiredClients": active["required_clients"]}


@router.post("/submit-update")
def submit_model_update(data: FLModelUpdate):
    if not client_exists(data.clientId):
        raise HTTPException(status_code=404, detail="FL client is not registered.")
    active = ensure_active_round()
    if data.roundId != int(active["round_id"]):
        raise HTTPException(status_code=400, detail="Client submitted an update for an inactive FL round.")
    if not data.modelUpdate:
        raise HTTPException(status_code=400, detail="Model update cannot be empty.")
    if not all(isinstance(value, (int, float)) for value in data.modelUpdate.values()):
        raise HTTPException(status_code=400, detail="All model update values must be numeric.")
    saved = save_model_update(data.clientId, data.roundId, data.sampleCount, {k: float(v) for k, v in data.modelUpdate.items()}, data.metadata)
    if not saved:
        raise HTTPException(status_code=409, detail="This client already submitted an update for this round.")
    return {"success": True, "message": "Model update received.", "roundId": data.roundId, "aggregation": try_aggregate_round(data.roundId)}


@router.post("/start-round")
def start_fl_round(data: FLRoundRequest):
    active = ensure_active_round(data.requiredClients)
    return {"success": True, "round": active}


# ===================== REAL RIDE DATA PIPELINE =====================

@router.post("/ride-data/submit")
def submit_real_ride_data(data: RideDatasetUpload):
    if not client_exists(data.clientId):
        raise HTTPException(status_code=404, detail="FL client is not registered. Register the device before submitting ride data.")
    valid_records, summary = validate_records(data.records)
    if not valid_records:
        raise HTTPException(status_code=400, detail="No valid sensor records were found in this ride dataset.")
    feature_row = build_ride_feature_row(valid_records)
    readiness = dataset_readiness(valid_records)
    paths = save_ride_dataset(data.clientId, data.rideId, valid_records, feature_row, summary, readiness, data.metadata)
    return {
        "success": True,
        "clientId": data.clientId,
        "rideId": data.rideId,
        "summary": summary,
        "readiness": readiness,
        "featurePreview": feature_row,
        "storage": paths,
        "message": "Ride data accepted and converted into a training-ready feature record. The production RideGuardian model has not been modified."
    }


@router.get("/ride-data/client/{client_id}")
def get_client_ride_data(client_id: str):
    if not client_exists(client_id):
        raise HTTPException(status_code=404, detail="FL client is not registered.")
    datasets = get_client_datasets(client_id)
    return {"success": True, "clientId": client_id, "rides": datasets, "rideCount": len(datasets)}


@router.get("/ride-data/stats")
def ride_dataset_stats():
    stats = get_all_dataset_stats()
    return {"success": True, "dataset": stats, "note": "These are collected FL training-pipeline datasets. They have not yet retrained the production model."}


@router.post("/ride-data/export-features")
def export_client_features(data: DatasetExportRequest):
    if not client_exists(data.clientId):
        raise HTTPException(status_code=404, detail="FL client is not registered.")
    rows = load_client_feature_rows(data.clientId)
    if not data.includeLabels:
        for row in rows:
            row.pop("riskLabel", None)
            row.pop("labelCoveragePercent", None)
    return JSONResponse({
        "success": True,
        "clientId": data.clientId,
        "samples": rows,
        "sampleCount": len(rows),
        "message": "Feature rows exported for the future local training stage."
    })
