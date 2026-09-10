import math
import statistics
from typing import Any, Dict, List, Tuple


SENSOR_FIELDS = [
    "speedKmh",
    "accelerationX",
    "accelerationY",
    "accelerationZ",
    "gyroX",
    "gyroY",
    "gyroZ",
    "timeSincePreviousReadingSec",
    "speedChangeKmh",
    "speedAccelerationMs2",
    "gyroMagnitudeRadS",
]

GPS_FIELDS = ["latitude", "longitude"]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _record_to_dict(record: Any) -> Dict[str, Any]:
    if hasattr(record, "model_dump"):
        return record.model_dump()
    if hasattr(record, "dict"):
        return record.dict()
    return dict(record)


def validate_records(records: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    valid_records: List[Dict[str, Any]] = []
    rejected = 0
    field_counts = {field: 0 for field in SENSOR_FIELDS}
    labelled = 0

    for raw in records:
        record = _record_to_dict(raw)
        cleaned: Dict[str, Any] = {
            "timestamp": record.get("timestamp"),
            "riskLabel": record.get("riskLabel"),
        }

        valid_sensor_count = 0

        for field in SENSOR_FIELDS:
            value = record.get(field)
            if _finite(value):
                cleaned[field] = float(value)
                valid_sensor_count += 1
                field_counts[field] += 1
            else:
                cleaned[field] = None

        for field in GPS_FIELDS:
            value = record.get(field)
            cleaned[field] = float(value) if _finite(value) else None

        if cleaned["riskLabel"] is not None:
            labelled += 1

        # A record with no useful sensor information is not training data.
        if valid_sensor_count == 0:
            rejected += 1
            continue

        valid_records.append(cleaned)

    total_input = len(records)
    valid_count = len(valid_records)
    completeness = {
        field: round((count / valid_count) * 100, 2) if valid_count else 0.0
        for field, count in field_counts.items()
    }

    return valid_records, {
        "inputRecords": total_input,
        "acceptedRecords": valid_count,
        "rejectedRecords": rejected,
        "labelledRecords": labelled,
        "fieldCompletenessPercent": completeness,
    }


def _numeric_values(records: List[Dict[str, Any]], field: str) -> List[float]:
    return [float(record[field]) for record in records if _finite(record.get(field))]


def build_ride_feature_row(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("Cannot build training features from an empty ride.")

    row: Dict[str, Any] = {"recordCount": len(records)}

    for field in SENSOR_FIELDS:
        values = _numeric_values(records, field)
        if not values:
            row[f"{field}_mean"] = None
            row[f"{field}_std"] = None
            row[f"{field}_min"] = None
            row[f"{field}_max"] = None
            continue

        row[f"{field}_mean"] = round(statistics.fmean(values), 6)
        row[f"{field}_std"] = round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0
        row[f"{field}_min"] = round(min(values), 6)
        row[f"{field}_max"] = round(max(values), 6)

    speed_acc = _numeric_values(records, "speedAccelerationMs2")
    gyro = _numeric_values(records, "gyroMagnitudeRadS")
    speed = _numeric_values(records, "speedKmh")

    row["harshBrakeRatio"] = round(
        sum(value < -2.5 for value in speed_acc) / len(speed_acc), 6
    ) if speed_acc else 0.0

    row["hardAccelerationRatio"] = round(
        sum(value > 2.5 for value in speed_acc) / len(speed_acc), 6
    ) if speed_acc else 0.0

    row["sharpTurnRatio"] = round(
        sum(abs(value) > 2.5 for value in gyro) / len(gyro), 6
    ) if gyro else 0.0

    row["highSpeedRatio"] = round(
        sum(value >= 70 for value in speed) / len(speed), 6
    ) if speed else 0.0

    labels = [record.get("riskLabel") for record in records if record.get("riskLabel") in (0, 1)]
    if labels:
        # Majority ride label is used only as a dataset-preparation label.
        row["riskLabel"] = 1 if (sum(labels) / len(labels)) >= 0.5 else 0
        row["labelCoveragePercent"] = round((len(labels) / len(records)) * 100, 2)
    else:
        row["riskLabel"] = None
        row["labelCoveragePercent"] = 0.0

    return row


def dataset_readiness(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "ready": False,
            "score": 0,
            "reason": "No accepted sensor records available.",
        }

    feature_completeness = []
    for field in SENSOR_FIELDS:
        count = sum(_finite(record.get(field)) for record in records)
        feature_completeness.append(count / len(records))

    completeness_score = statistics.fmean(feature_completeness) * 70
    volume_score = min(len(records) / 5000, 1.0) * 20
    labels = sum(record.get("riskLabel") in (0, 1) for record in records)
    label_score = min(labels / max(len(records), 1), 1.0) * 10
    score = round(completeness_score + volume_score + label_score, 2)

    return {
        "ready": len(records) >= 100 and statistics.fmean(feature_completeness) >= 0.70,
        "score": score,
        "acceptedRecords": len(records),
        "averageFeatureCompletenessPercent": round(statistics.fmean(feature_completeness) * 100, 2),
        "labelledRecords": labels,
        "note": "Readiness means the dataset is structurally suitable for the future training stage; it does not mean the production model has been retrained.",
    }
