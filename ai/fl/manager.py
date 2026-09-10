from threading import Lock
from typing import Any, Dict

from .aggregator import federated_average

from .storage import (
    complete_round,
    create_round,
    get_global_model,
    get_latest_round,
    get_registered_client_count,
    get_round,
    get_round_update_count,
    get_round_updates,
    save_global_model,
)


FL_MANAGER_LOCK = Lock()


def get_current_model_version() -> int:
    return int(get_global_model().get("version", 1))


def ensure_active_round(
    required_clients: int = 3
) -> Dict[str, Any]:

    with FL_MANAGER_LOCK:

        latest = get_latest_round()

        if latest is None:

            create_round(
                1,
                get_current_model_version(),
                required_clients
            )

            return get_round(1)

        if latest["status"] == "WAITING_FOR_UPDATES":

            return latest

        new_id = int(
            latest["round_id"]
        ) + 1

        create_round(
            new_id,
            get_current_model_version(),
            required_clients
        )

        return get_round(new_id)


def try_aggregate_round(
    round_id: int
) -> Dict[str, Any]:

    with FL_MANAGER_LOCK:

        current = get_round(round_id)

        if current is None:

            return {
                "aggregated": False,
                "reason": "Round does not exist."
            }

        if current["status"] != "WAITING_FOR_UPDATES":

            return {
                "aggregated": False,
                "reason": "Round is no longer active."
            }

        submitted = get_round_update_count(
            round_id
        )

        required = int(
            current["required_clients"]
        )

        if submitted < required:

            return {
                "aggregated": False,
                "reason":
                    "Waiting for more client updates.",
                "submittedUpdates":
                    submitted,
                "requiredClients":
                    required
            }

        # ====================================================
        # GET CLIENT UPDATES
        # ====================================================

        updates = get_round_updates(
            round_id
        )

        client_updates = [
            update["model_update"]
            for update in updates
        ]

        client_sample_counts = [
            int(update["sample_count"])
            for update in updates
        ]

        # ====================================================
        # FEDERATED AVERAGING
        # ====================================================

        global_parameters = federated_average(
            client_updates,
            client_sample_counts
        )

        total_samples = sum(
            client_sample_counts
        )

        client_count = len(
            client_updates
        )

        # ====================================================
        # CREATE NEW GLOBAL MODEL VERSION
        # ====================================================

        current_model = get_global_model()

        new_version = (
            int(
                current_model.get(
                    "version",
                    1
                )
            ) + 1
        )

        save_global_model(
            new_version,
            {
                "parameters":
                    global_parameters,

                "totalSamples":
                    total_samples,

                "aggregatedClients":
                    client_count,

                "sourceRound":
                    round_id,

                "status":
                    "aggregated_pipeline_model"
            },
            round_id
        )

        # ====================================================
        # COMPLETE ROUND
        # ====================================================

        complete_round(
            round_id
        )

        return {
            "aggregated": True,
            "newModelVersion":
                new_version,
            "clientsUsed":
                client_count,
            "totalSamples":
                total_samples
        }


def get_fl_status() -> Dict[str, Any]:

    active = ensure_active_round()

    model = get_global_model()

    return {
        "currentRound":
            int(active["round_id"]),

        "modelVersion":
            int(
                model.get(
                    "version",
                    1
                )
            ),

        "status":
            active["status"],

        "registeredClients":
            get_registered_client_count(),

        "submittedUpdates":
            get_round_update_count(
                int(active["round_id"])
            ),

        "requiredClients":
            int(
                active["required_clients"]
            )
    }