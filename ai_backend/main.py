# ============================================================
# RIDEGUARDIAN AI BACKEND
# ============================================================

import sys

from pathlib import Path

from fastapi import FastAPI


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=
    "RideGuardian AI Backend",

    description=
    "Vehicle AI, Tyre Analysis and Federated Learning Backend",

    version=
    "2.0.0"

)


# ============================================================
# IMPORT ROUTERS
# ============================================================

from ai.api.vehicle_diagnosis_api import (
    router as vehicle_router
)


# ============================================================
# REGISTER ROUTERS
# ============================================================

app.include_router(

    vehicle_router

)


# ============================================================
# HOME
# ============================================================

@app.get("/")

def home():

    return {

        "message":
        "RideGuardian AI Backend is running",

        "status":
        "online",

        "services": [

            "Vehicle Parts Detection",

            "Vehicle Damage Detection",

            "Tyre Condition Analysis"

        ]

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")

def health():

    return {

        "status":
        "healthy",

        "backend":
        "RideGuardian AI Backend"

    }