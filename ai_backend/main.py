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
    "Vehicle AI, Tyre Analysis, Route Map Matching "
    "and Federated Learning Backend",

    version=
    "2.0.0"

)


# ============================================================
# IMPORT ROUTERS
# ============================================================

from ai.api.vehicle_diagnosis_api import (
    router as vehicle_router
)

from ai.api.route_api import (
    router as route_router
)

from ai.fl.storage import (
    initialize_fl_database
)

from ai.fl.routes import (
    router as fl_router
)


# ============================================================
# INITIALIZE FEDERATED LEARNING DATABASE
# ============================================================

initialize_fl_database()


# ============================================================
# REGISTER ROUTERS
# ============================================================

app.include_router(

    vehicle_router

)

app.include_router(

    route_router

)

app.include_router(

    fl_router

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

            "Tyre Condition Analysis",

            "Route Map Matching",

            "Federated Learning"

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