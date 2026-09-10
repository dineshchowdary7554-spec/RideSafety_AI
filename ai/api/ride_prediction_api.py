# ============================================================
# RIDEGUARDIAN LIVE RIDE AI API
# ============================================================
#
# Receives the latest mobile sensor sample and performs the
# backend-side live ride analysis.
#
# Responsibilities:
#   - Riding-risk prediction
#   - Vehicle wear/stress estimation
#   - Thermal/temperature estimation
#
# IMPORTANT:
# Mobile remains responsible for collecting GPS, accelerometer
# and gyroscope data. The backend is responsible for prediction.
#
# There is currently no trained engine-temperature model in the
# supplied project files. Therefore the temperature returned here
# is explicitly an ESTIMATE based on driving load. If an OBD
# coolant-temperature value is supplied in the future, that real
# value takes priority.
# ============================================================

import math
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter(
    tags=["Live Ride AI"]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class RidePredictionRequest(BaseModel):
    speedKmh: float = 0.0

    accelerationX: float = 0.0
    accelerationY: float = 0.0
    accelerationZ: float = 0.0

    gyroX: float = 0.0
    gyroY: float = 0.0
    gyroZ: float = 0.0

    timeSincePreviousReadingSec: float = 5.0
    speedChangeKmh: float = 0.0
    speedAccelerationMs2: float = 0.0
    gyroMagnitudeRadS: float = 0.0

    linearAccelerationMagnitudeMs2: Optional[float] = None

    # Optional future OBD input.
    obdCoolantTemperatureC: Optional[float] = None


# ============================================================
# NUMBER HELPERS
# ============================================================

def safe_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    return max(
        minimum,
        min(maximum, value)
    )


def rounded(value: float, digits: int = 2) -> float:
    return round(
        safe_number(value),
        digits
    )


# ============================================================
# CONDITION HELPERS
# ============================================================

def get_stress_condition(
    overall_stress: float,
) -> str:

    if overall_stress < 25:
        return "LOW_STRESS"

    if overall_stress < 50:
        return "MODERATE_STRESS"

    if overall_stress < 75:
        return "HIGH_STRESS"

    return "SEVERE_STRESS"


def get_temperature_status(
    temperature_c: float,
) -> str:

    if temperature_c < 80:
        return "NORMAL"

    if temperature_c < 90:
        return "ELEVATED"

    if temperature_c < 100:
        return "HIGH"

    return "VERY_HIGH"


# ============================================================
# SENSOR FEATURE CALCULATION
# ============================================================

def calculate_sensor_features(
    data: RidePredictionRequest,
) -> Dict[str, float]:

    acceleration_magnitude_g = math.sqrt(
        safe_number(data.accelerationX) ** 2
        + safe_number(data.accelerationY) ** 2
        + safe_number(data.accelerationZ) ** 2
    )

    gyro_magnitude = math.sqrt(
        safe_number(data.gyroX) ** 2
        + safe_number(data.gyroY) ** 2
        + safe_number(data.gyroZ) ** 2
    )

    if data.gyroMagnitudeRadS is not None:
        supplied_gyro = safe_number(
            data.gyroMagnitudeRadS
        )

        if supplied_gyro > 0:
            gyro_magnitude = supplied_gyro

    if (
        data.linearAccelerationMagnitudeMs2
        is not None
        and safe_number(
            data.linearAccelerationMagnitudeMs2
        ) > 0
    ):
        linear_acceleration_ms2 = safe_number(
            data.linearAccelerationMagnitudeMs2
        )

    else:
        # The mobile app sends cleaned linear acceleration
        # values in approximately g units.
        linear_acceleration_ms2 = (
            acceleration_magnitude_g * 9.80665
        )

    return {
        "accelerationMagnitudeG":
            acceleration_magnitude_g,

        "gyroMagnitudeRadS":
            gyro_magnitude,

        "linearAccelerationMagnitudeMs2":
            linear_acceleration_ms2,

        "speedKmh":
            max(0.0, safe_number(data.speedKmh)),

        "speedChangeKmh":
            safe_number(data.speedChangeKmh),

        "speedAccelerationMs2":
            safe_number(data.speedAccelerationMs2),

        "timeSincePreviousReadingSec":
            max(
                0.1,
                safe_number(
                    data.timeSincePreviousReadingSec,
                    5.0,
                ),
            ),
    }


# ============================================================
# LIVE RISK PREDICTION
# ============================================================

def calculate_risk(
    features: Dict[str, float],
) -> Dict[str, Any]:

    speed = features["speedKmh"]
    acceleration = features["speedAccelerationMs2"]
    gyro = features["gyroMagnitudeRadS"]
    linear_acceleration = features[
        "linearAccelerationMagnitudeMs2"
    ]

    # These thresholds follow the same general safety-event
    # thresholds already used by the mobile safety-score system.
    speed_risk = 0.0

    if speed > 80:
        speed_risk = min(
            35.0,
            (speed - 80.0) * 0.70
        )

    hard_acceleration_excess = max(
        0.0,
        acceleration - 2.5
    )

    hard_braking_excess = max(
        0.0,
        -acceleration - 2.5
    )

    sudden_turn_excess = max(
        0.0,
        gyro - 1.2
    )

    vibration_excess = max(
        0.0,
        linear_acceleration - 2.0
    )

    acceleration_risk = min(
        25.0,
        hard_acceleration_excess * 12.0
    )

    braking_risk = min(
        30.0,
        hard_braking_excess * 14.0
    )

    cornering_risk = min(
        20.0,
        sudden_turn_excess * 18.0
    )

    vibration_risk = min(
        15.0,
        vibration_excess * 6.0
    )

    risk_probability = clamp(
        speed_risk
        + acceleration_risk
        + braking_risk
        + cornering_risk
        + vibration_risk
    )

    prediction = (
        "RISK_EVENT"
        if risk_probability >= 35.0
        else "NORMAL"
    )

    reasons = []

    if speed_risk >= 10:
        reasons.append("high speed")

    if acceleration_risk >= 8:
        reasons.append("hard acceleration")

    if braking_risk >= 8:
        reasons.append("hard braking")

    if cornering_risk >= 8:
        reasons.append("high cornering load")

    if vibration_risk >= 6:
        reasons.append("high vibration")

    return {
        "prediction": prediction,
        "riskProbability": rounded(
            risk_probability
        ),
        "reasons": reasons,
    }


# ============================================================
# VEHICLE WEAR / STRESS PREDICTION
# ============================================================

def calculate_vehicle_wear(
    features: Dict[str, float],
) -> Dict[str, Any]:

    speed = features["speedKmh"]
    acceleration = features["speedAccelerationMs2"]
    gyro = features["gyroMagnitudeRadS"]
    linear_acceleration = features[
        "linearAccelerationMagnitudeMs2"
    ]

    speed_load = max(
        0.0,
        speed - 60.0
    )

    high_speed_load = max(
        0.0,
        speed - 80.0
    )

    positive_acceleration = max(
        0.0,
        acceleration
    )

    braking_excess = max(
        0.0,
        -acceleration - 2.5
    )

    cornering_excess = max(
        0.0,
        gyro - 1.2
    )

    vibration_excess = max(
        0.0,
        linear_acceleration - 2.0
    )

    # --------------------------------------------------------
    # TYRE STRESS
    # --------------------------------------------------------

    tyre_wear_risk = clamp(
        speed_load * 0.25
        + cornering_excess * 22.0
        + positive_acceleration * 4.0
        + max(0.0, -acceleration) * 3.0
    )

    # --------------------------------------------------------
    # BRAKE STRESS
    # --------------------------------------------------------

    brake_wear_risk = clamp(
        braking_excess * 18.0
        + high_speed_load * 0.25
    )

    # --------------------------------------------------------
    # SUSPENSION STRESS
    # --------------------------------------------------------

    suspension_stress = clamp(
        vibration_excess * 18.0
        + cornering_excess * 12.0
        + max(0.0, speed - 70.0) * 0.15
    )

    # --------------------------------------------------------
    # DRIVETRAIN STRESS
    # --------------------------------------------------------

    drivetrain_stress = clamp(
        positive_acceleration * 18.0
        + max(0.0, speed - 70.0) * 0.20
    )

    # --------------------------------------------------------
    # OVERALL VEHICLE STRESS
    # --------------------------------------------------------

    overall_stress = clamp(
        brake_wear_risk * 0.25
        + tyre_wear_risk * 0.30
        + suspension_stress * 0.25
        + drivetrain_stress * 0.20
    )

    condition = get_stress_condition(
        overall_stress
    )

    return {
        "overallStress":
            rounded(overall_stress),

        "condition":
            condition,

        "tyreWearRisk":
            rounded(tyre_wear_risk),

        "brakeWearRisk":
            rounded(brake_wear_risk),

        "suspensionStress":
            rounded(suspension_stress),

        "drivetrainStress":
            rounded(drivetrain_stress),
    }


# ============================================================
# TEMPERATURE / THERMAL ESTIMATION
# ============================================================

def calculate_temperature(
    data: RidePredictionRequest,
    features: Dict[str, float],
    vehicle_wear: Dict[str, Any],
) -> Dict[str, Any]:

    # If OBD supplies a real coolant temperature, use it.
    obd_temperature = data.obdCoolantTemperatureC

    if (
        obd_temperature is not None
        and math.isfinite(
            safe_number(obd_temperature)
        )
        and 20.0 <= safe_number(obd_temperature) <= 150.0
    ):
        temperature_c = safe_number(
            obd_temperature
        )

        return {
            "estimatedC":
                rounded(temperature_c, 1),

            "status":
                get_temperature_status(
                    temperature_c
                ),

            "source":
                "OBD_COOLANT_TEMPERATURE",
        }

    # No direct engine-temperature sensor/model was supplied.
    # This is therefore a conservative behavioural thermal-load
    # estimate, not a physical coolant-temperature measurement.
    speed = features["speedKmh"]
    acceleration = features["speedAccelerationMs2"]
    overall_stress = safe_number(
        vehicle_wear["overallStress"]
    )

    positive_acceleration = max(
        0.0,
        acceleration
    )

    estimated_temperature = clamp(
        65.0
        + speed * 0.08
        + overall_stress * 0.18
        + positive_acceleration * 1.5,
        55.0,
        105.0,
    )

    return {
        "estimatedC":
            rounded(
                estimated_temperature,
                1,
            ),

        "status":
            get_temperature_status(
                estimated_temperature
            ),

        "source":
            "BEHAVIOURAL_THERMAL_ESTIMATE",
    }


# ============================================================
# LIVE PREDICTION ENDPOINT
# ============================================================

@router.post(
    "/predict"
)
def predict_live_ride(
    data: RidePredictionRequest,
) -> Dict[str, Any]:

    features = calculate_sensor_features(
        data
    )

    risk = calculate_risk(
        features
    )

    vehicle_wear = calculate_vehicle_wear(
        features
    )

    temperature = calculate_temperature(
        data,
        features,
        vehicle_wear,
    )

    return {
        "success": True,

        "prediction":
            risk["prediction"],

        "riskProbability":
            risk["riskProbability"],

        "sensorAnalysis": {
            "accelerationMagnitude":
                rounded(
                    features[
                        "accelerationMagnitudeG"
                    ],
                    4,
                ),

            "gyroMagnitude":
                rounded(
                    features[
                        "gyroMagnitudeRadS"
                    ],
                    4,
                ),

            "linearAccelerationMagnitudeMs2":
                rounded(
                    features[
                        "linearAccelerationMagnitudeMs2"
                    ],
                    4,
                ),
        },

        "vehicleWear":
            vehicle_wear,

        "temperature":
            temperature,

        "analysisReasons":
            risk["reasons"],

        "backendVersion":
            "2.1.0",
    }
