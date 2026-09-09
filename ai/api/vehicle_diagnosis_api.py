# ============================================================
# RIDEGUARDIAN
# UNIFIED VEHICLE DIAGNOSIS AI API
#
# Models:
# 1. Vehicle Parts YOLO
# 2. Vehicle Damage YOLO
# 3. Tyre Condition EfficientNet-B0
#
# Workflow:
#
# Image
#   |
#   +--> Vehicle Parts YOLO
#   |        |
#   |        +--> Tyre detected
#   |        |       |
#   |        |       +--> Crop tyre
#   |        |               |
#   |        |               +--> Tyre Condition Model
#   |        |
#   |        +--> Other vehicle part
#   |
#   +--> Damage YOLO
#
# Final result:
# Combined RideGuardian diagnosis
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import io
import threading

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import timm

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile
)

from PIL import Image

from torchvision import transforms

from ultralytics import YOLO

# ============================================================
# TYRE CONDITION INTERPRETATION
# ============================================================

def get_tyre_health(condition: str, confidence: float):
    """
    Converts the AI tyre tread prediction into a user-friendly
    health score, condition level and recommendation.
    """

    tyre_data = {

        "8mm": {
            "health_score": 100,
            "condition_level": "Excellent",
            "status": "Excellent tyre condition",
            "recommendation":
                "Your tyre appears to be in excellent condition. "
                "No immediate action is required. Continue regular inspections."
        },

        "7mm": {
            "health_score": 92,
            "condition_level": "Very Good",
            "status": "Very good tyre condition",
            "recommendation":
                "Your tyre is in very good condition with sufficient tread. "
                "Continue normal maintenance and regular inspections."
        },

        "6.5mm": {
            "health_score": 85,
            "condition_level": "Good",
            "status": "Good tyre condition",
            "recommendation":
                "Your tyre is in good condition. Continue monitoring "
                "the tread during regular vehicle servicing."
        },

        "5mm": {
            "health_score": 68,
            "condition_level": "Moderate Wear",
            "status": "Moderate tyre wear detected",
            "recommendation":
                "Your tyre shows moderate wear. Continue monitoring "
                "the tread and plan an inspection during future servicing."
        },

        "3mm": {
            "health_score": 40,
            "condition_level": "Worn",
            "status": "Significant tyre wear detected",
            "recommendation":
                "Your tyre tread is significantly worn. A professional "
                "inspection is recommended and tyre replacement may soon be required."
        },

        "under_3mm": {
            "health_score": 15,
            "condition_level": "Critical",
            "status": "Critical tyre wear detected",
            "recommendation":
                "Your tyre tread appears critically worn. We strongly "
                "recommend immediate professional inspection and considering tyre replacement."
        }
    }

    result = tyre_data.get(
        condition,
        {
            "health_score": 50,
            "condition_level": "Unknown",
            "status": "Tyre condition detected",
            "recommendation":
                "Please inspect your tyre condition manually."
        }
    )

    return {
        "tyre_condition": condition,
        "tyre_confidence": round(float(confidence), 4),
        "tyre_confidence_percent": round(float(confidence) * 100, 2),
        "tyre_health_score": result["health_score"],
        "condition_level": result["condition_level"],
        "status": result["status"],
        "recommendation": result["recommendation"]
    }

# ============================================================
# CREATE ROUTER
# ============================================================

router = APIRouter(
    tags=["Vehicle AI"]
)


# ============================================================
# PROJECT ROOT
#
# File location:
#
# RideGuardian/
# ├── ai/
# │   ├── api/
# │   │   └── vehicle_diagnosis_api.py
# │   └── models/
# │
# └── ai_backend/
#
# Therefore:
#
# parents[0] = api
# parents[1] = ai
# parents[2] = RideGuardian
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


# ============================================================
# MODEL PATHS
# ============================================================


# ------------------------------------------------------------
# TYRE CONDITION MODEL
# ------------------------------------------------------------

TYRE_MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "tyre_condition_model.pth"
)


# ------------------------------------------------------------
# VEHICLE PARTS YOLO
#
# Based on your discovered model structure:
#
# ai/models/vehicle_parts_v2/
#     vehicle_parts_v2/
#         weights/
#             best.pt
# ------------------------------------------------------------

VEHICLE_PART_MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "vehicle_parts_v2"
    / "vehicle_parts_v2"
    / "weights"
    / "best.pt"
)


# ------------------------------------------------------------
# DAMAGE YOLO
# ------------------------------------------------------------

DAMAGE_MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "damage"
    / "damage_v1"
    / "weights"
    / "best.pt"
)


# ============================================================
# TYRE MODEL CONFIGURATION
#
# Confirmed from your checkpoint:
#
# Architecture: efficientnet_b0
# Classes: 6
# Image size: 224
# ============================================================

TYRE_ARCHITECTURE = "efficientnet_b0"

TYRE_IMAGE_SIZE = 224

TYRE_CLASSES = [
    "3mm",
    "5mm",
    "6.5mm",
    "7mm",
    "8mm",
    "under_3mm"
]


# ============================================================
# CONFIDENCE SETTINGS
# ============================================================

# Minimum confidence for vehicle part detection

PART_CONFIDENCE = 0.35


# Minimum confidence for damage detection

DAMAGE_CONFIDENCE = 0.50


# A close-up tyre image is accepted as a tyre only when the
# tyre classification model is sufficiently confident.

TYRE_FALLBACK_CONFIDENCE = 0.70


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 70)
print("RIDEGUARDIAN UNIFIED VEHICLE AI")
print("=" * 70)

print(
    "Device:",
    DEVICE
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print("=" * 70)


# ============================================================
# CHECK MODEL FILES
# ============================================================

REQUIRED_MODEL_FILES = {
    "Tyre Condition Model":
        TYRE_MODEL_PATH,

    "Vehicle Parts YOLO":
        VEHICLE_PART_MODEL_PATH,

    "Damage YOLO":
        DAMAGE_MODEL_PATH
}


for model_name, model_path in REQUIRED_MODEL_FILES.items():

    if model_path.exists():

        print(
            f"FOUND: {model_name}"
        )

        print(
            model_path
        )

    else:

        print(
            f"WARNING: {model_name} NOT FOUND"
        )

        print(
            model_path
        )


# ============================================================
# IMAGE TRANSFORM
#
# Must match tyre model training preprocessing
# ============================================================

tyre_transform = transforms.Compose([

    transforms.Resize(
        (
            TYRE_IMAGE_SIZE,
            TYRE_IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# GLOBAL MODEL VARIABLES
# ============================================================

vehicle_part_model = None

damage_model = None

tyre_model = None


# ============================================================
# INFERENCE LOCK
#
# Prevents multiple simultaneous model inferences from causing
# excessive CPU/GPU/RAM usage.
# ============================================================

inference_lock = threading.Lock()


# ============================================================
# LOAD VEHICLE PART YOLO MODEL
# ============================================================

def load_vehicle_part_model():

    print()
    print("=" * 70)
    print("LOADING VEHICLE PARTS YOLO")
    print("=" * 70)

    if not VEHICLE_PART_MODEL_PATH.exists():

        print(
            "Vehicle Parts YOLO model was not found."
        )

        return None

    try:

        model = YOLO(
            str(
                VEHICLE_PART_MODEL_PATH
            )
        )

        print(
            "Vehicle Parts YOLO loaded successfully."
        )

        print(
            "Classes:",
            model.names
        )

        return model

    except Exception as error:

        print(
            "ERROR LOADING VEHICLE PART YOLO:"
        )

        print(
            repr(error)
        )

        return None


# ============================================================
# LOAD DAMAGE YOLO MODEL
# ============================================================

def load_damage_model():

    print()
    print("=" * 70)
    print("LOADING DAMAGE YOLO")
    print("=" * 70)

    if not DAMAGE_MODEL_PATH.exists():

        print(
            "Damage YOLO model was not found."
        )

        return None

    try:

        model = YOLO(
            str(
                DAMAGE_MODEL_PATH
            )
        )

        print(
            "Damage YOLO loaded successfully."
        )

        print(
            "Classes:",
            model.names
        )

        return model

    except Exception as error:

        print(
            "ERROR LOADING DAMAGE YOLO:"
        )

        print(
            repr(error)
        )

        return None


# ============================================================
# LOAD TYRE CONDITION MODEL
# ============================================================

def load_tyre_model():

    print()
    print("=" * 70)
    print("LOADING TYRE CONDITION MODEL")
    print("=" * 70)

    if not TYRE_MODEL_PATH.exists():

        print(
            "Tyre condition model was not found."
        )

        return None

    try:

        # ----------------------------------------------------
        # CREATE EXACT TRAINING ARCHITECTURE
        # ----------------------------------------------------

        model = timm.create_model(

            TYRE_ARCHITECTURE,

            pretrained=False,

            num_classes=len(
                TYRE_CLASSES
            )
        )


        # ----------------------------------------------------
        # LOAD CHECKPOINT
        # ----------------------------------------------------

        checkpoint = torch.load(

            TYRE_MODEL_PATH,

            map_location=DEVICE,

            weights_only=False
        )


        # ----------------------------------------------------
        # CHECKPOINT STRUCTURE
        # ----------------------------------------------------

        if isinstance(
            checkpoint,
            dict
        ):

            if (
                "model_state_dict"
                in checkpoint
            ):

                state_dict = (
                    checkpoint[
                        "model_state_dict"
                    ]
                )

            elif (
                "state_dict"
                in checkpoint
            ):

                state_dict = (
                    checkpoint[
                        "state_dict"
                    ]
                )

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint


        # ----------------------------------------------------
        # REMOVE DDP PREFIX IF PRESENT
        # ----------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith(
                "module."
            ):

                new_key = new_key.replace(
                    "module.",
                    "",
                    1
                )

            cleaned_state_dict[
                new_key
            ] = value


        # ----------------------------------------------------
        # LOAD MODEL WEIGHTS
        # ----------------------------------------------------

        model.load_state_dict(
            cleaned_state_dict
        )

        model.to(
            DEVICE
        )

        model.eval()


        # ----------------------------------------------------
        # PRINT CHECKPOINT INFORMATION
        # ----------------------------------------------------

        print(
            "Tyre model loaded successfully."
        )

        if isinstance(
            checkpoint,
            dict
        ):

            print()

            print(
                "Checkpoint architecture:",
                checkpoint.get(
                    "architecture",
                    "Unknown"
                )
            )

            print(
                "Checkpoint classes:",
                checkpoint.get(
                    "class_names",
                    TYRE_CLASSES
                )
            )

            print(
                "Checkpoint image size:",
                checkpoint.get(
                    "image_size",
                    TYRE_IMAGE_SIZE
                )
            )

        return model

    except Exception as error:

        print()
        print(
            "ERROR LOADING TYRE CONDITION MODEL:"
        )

        print(
            repr(error)
        )

        return None


# ============================================================
# LOAD ALL MODELS
# ============================================================

vehicle_part_model = (
    load_vehicle_part_model()
)

damage_model = (
    load_damage_model()
)

tyre_model = (
    load_tyre_model()
)


print()
print("=" * 70)
print("RIDEGUARDIAN MODEL STATUS")
print("=" * 70)

print(
    "Vehicle Parts YOLO:",
    "READY"
    if vehicle_part_model
    else "NOT AVAILABLE"
)

print(
    "Damage YOLO:",
    "READY"
    if damage_model
    else "NOT AVAILABLE"
)

print(
    "Tyre Condition Model:",
    "READY"
    if tyre_model
    else "NOT AVAILABLE"
)

print("=" * 70)


# ============================================================
# HELPER:
# NORMALIZE YOLO CLASS NAME
# ============================================================

def normalize_class_name(
    name: str
) -> str:

    return (

        str(name)

        .lower()

        .replace(
            "_",
            " "
        )

        .replace(
            "-",
            " "
        )

        .strip()
    )


# ============================================================
# HELPER:
# CHECK IF PART IS A TYRE
# ============================================================

def is_tyre_part(
    part_name: str
) -> bool:

    normalized_name = (
        normalize_class_name(
            part_name
        )
    )

    tyre_keywords = [

        "tyre",

        "tire",

        "wheel"
    ]

    return any(

        keyword
        in normalized_name

        for keyword
        in tyre_keywords
    )


# ============================================================
# HELPER:
# EXTRACT YOLO DETECTIONS
#
# Includes bounding boxes because we need to crop tyres.
# ============================================================

def extract_yolo_detections(
    results,
    max_predictions=10
) -> List[Dict[str, Any]]:

    detections = []

    if not results:

        return detections


    result = results[0]


    if result.boxes is None:

        return detections


    names = result.names


    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )


        class_name = str(

            names[
                class_id
            ]
        )


        coordinates = (

            box.xyxy[0]

            .detach()

            .cpu()

            .tolist()
        )


        x1, y1, x2, y2 = coordinates


        detections.append({

            "class":
                class_name,

            "confidence":
                round(
                    confidence,
                    6
                ),

            "confidence_percent":
                round(
                    confidence * 100,
                    2
                ),

            "bbox": {

                "x1":
                    round(x1, 2),

                "y1":
                    round(y1, 2),

                "x2":
                    round(x2, 2),

                "y2":
                    round(y2, 2)
            }
        })


    detections = sorted(

        detections,

        key=lambda item:
            item["confidence"],

        reverse=True
    )


    return detections[
        :max_predictions
    ]


# ============================================================
# DETECT VEHICLE PARTS
# ============================================================

def detect_vehicle_parts(
    image: Image.Image
) -> List[Dict[str, Any]]:

    if vehicle_part_model is None:

        return []


    results = vehicle_part_model(

        image,

        conf=PART_CONFIDENCE,

        verbose=False
    )


    return extract_yolo_detections(
        results
    )


# ============================================================
# DETECT VEHICLE DAMAGE
# ============================================================

def detect_vehicle_damage(
    image: Image.Image
) -> List[Dict[str, Any]]:

    if damage_model is None:

        return []


    results = damage_model(

        image,

        conf=DAMAGE_CONFIDENCE,

        verbose=False
    )


    return extract_yolo_detections(
        results
    )


# ============================================================
# FIND BEST TYRE DETECTION
# ============================================================

def find_best_tyre_detection(
    part_detections
) -> Optional[Dict[str, Any]]:

    tyre_detections = []


    for detection in part_detections:

        if is_tyre_part(

            detection["class"]

        ):

            tyre_detections.append(
                detection
            )


    if not tyre_detections:

        return None


    tyre_detections = sorted(

        tyre_detections,

        key=lambda item:
            item["confidence"],

        reverse=True
    )


    return tyre_detections[0]


# ============================================================
# CROP DETECTED TYRE
# ============================================================

def crop_tyre_region(

    image: Image.Image,

    detection: Dict[str, Any]

) -> Image.Image:

    width, height = image.size

    bbox = detection["bbox"]


    x1 = max(
        0,
        int(
            bbox["x1"]
        )
    )

    y1 = max(
        0,
        int(
            bbox["y1"]
        )
    )

    x2 = min(
        width,
        int(
            bbox["x2"]
        )
    )

    y2 = min(
        height,
        int(
            bbox["y2"]
        )
    )


    if x2 <= x1 or y2 <= y1:

        raise ValueError(
            "Invalid tyre bounding box."
        )


    return image.crop(

        (
            x1,
            y1,
            x2,
            y2
        )
    )


# ============================================================
# TYRE CONDITION PREDICTION
# ============================================================

def predict_tyre_condition(

    image: Image.Image

) -> Dict[str, Any]:

    if tyre_model is None:

        raise RuntimeError(

            "Tyre condition model "
            "is not available."
        )


    image = image.convert(
        "RGB"
    )


    image_tensor = tyre_transform(
        image
    )


    image_tensor = (

        image_tensor

        .unsqueeze(0)

        .to(DEVICE)
    )


    with torch.inference_mode():

        outputs = tyre_model(
            image_tensor
        )


        probabilities = (

            torch.softmax(

                outputs,

                dim=1

            )[0]
        )


    confidence, predicted_index = (

        torch.max(

            probabilities,

            dim=0
        )
    )


    predicted_index = int(
        predicted_index.item()
    )

    confidence = float(
        confidence.item()
    )


    condition = TYRE_CLASSES[
        predicted_index
    ]


    # --------------------------------------------------------
    # TOP 3 PREDICTIONS
    # --------------------------------------------------------

    top_values, top_indices = (

        torch.topk(

            probabilities,

            k=min(
                3,
                len(TYRE_CLASSES)
            )
        )
    )


    top_predictions = []


    for value, index in zip(

        top_values,

        top_indices
    ):

        probability = float(
            value.item()
        )


        class_name = TYRE_CLASSES[

            int(
                index.item()
            )
        ]


        top_predictions.append({

            "class":
                class_name,

            "confidence":
                round(
                    probability,
                    6
                ),

            "confidence_percent":
                round(
                    probability * 100,
                    2
                )
        })


    return {

        "condition":
            condition,

        "confidence":
            round(
                confidence,
                6
            ),

        "confidence_percent":
            round(
                confidence * 100,
                2
            ),

        "top_predictions":
            top_predictions
    }


# ============================================================
# TYRE HEALTH SCORE
# ============================================================

def get_tyre_health_score(
    condition: str
) -> int:

    scores = {

        "8mm": 100,

        "7mm": 92,

        "6.5mm": 85,

        "5mm": 70,

        "3mm": 45,

        "under_3mm": 20
    }


    return scores.get(
        condition,
        50
    )


# ============================================================
# TYRE STATUS
# ============================================================

def get_tyre_status(
    condition: str
) -> str:

    status = {

        "8mm":
            "Excellent",

        "7mm":
            "Very Good",

        "6.5mm":
            "Good",

        "5mm":
            "Moderate Wear",

        "3mm":
            "High Wear",

        "under_3mm":
            "Critical Wear"
    }


    return status.get(
        condition,
        "Unknown"
    )


# ============================================================
# TYRE RECOMMENDATION
# ============================================================

def get_tyre_recommendation(
    condition: str
) -> str:

    recommendations = {

        "8mm":

            "Tyre appears to be in excellent condition "
            "with strong remaining tread.",


        "7mm":

            "Tyre condition appears very good. "
            "Continue normal maintenance and monitoring.",


        "6.5mm":

            "Tyre remains in good condition with moderate wear. "
            "Continue monitoring tread depth regularly.",


        "5mm":

            "Tyre shows noticeable wear. Continue monitoring "
            "tread depth and inspect the tyre regularly.",


        "3mm":

            "Tyre tread is significantly worn. A professional "
            "inspection and replacement planning are recommended.",


        "under_3mm":

            "Tyre tread appears critically worn. Replacement "
            "should be considered urgently for safety."
    }


    return recommendations.get(

        condition,

        "Tyre condition could not be interpreted."
    )


# ============================================================
# CONFIDENCE LEVEL
# ============================================================

def get_confidence_level(
    confidence: float
) -> str:

    if confidence >= 0.90:

        return "Very High"

    elif confidence >= 0.75:

        return "High"

    elif confidence >= 0.55:

        return "Medium"

    else:

        return "Low"


# ============================================================
# CREATE COMBINED VEHICLE RECOMMENDATION
# ============================================================

def generate_recommendation(

    tyre_prediction:

        Optional[
            Dict[str, Any]
        ],

    best_part:

        Optional[
            Dict[str, Any]
        ],

    best_damage:

        Optional[
            Dict[str, Any]
        ]

) -> str:


    # --------------------------------------------------------
    # TYRE DIAGNOSIS TAKES PRIORITY
    # --------------------------------------------------------

    if tyre_prediction:

        return get_tyre_recommendation(

            tyre_prediction[
                "condition"
            ]
        )


    # --------------------------------------------------------
    # DAMAGE + PART
    # --------------------------------------------------------

    if best_part and best_damage:

        return (

            f"{best_damage['class']} was detected "
            f"on the {best_part['class']}. "
            "A closer physical inspection is recommended."
        )


    # --------------------------------------------------------
    # PART ONLY
    # --------------------------------------------------------

    if best_part:

        return (

            f"The {best_part['class']} was detected, "
            "but no confident damage diagnosis was found."
        )


    # --------------------------------------------------------
    # DAMAGE ONLY
    # --------------------------------------------------------

    if best_damage:

        return (

            f"{best_damage['class']} was detected. "
            "Please inspect the affected vehicle area."
        )


    return (

        "Unable to confidently identify the vehicle component. "
        "Please upload a clearer image or provide more details."
    )


# ============================================================
# UNIFIED INFERENCE
# ============================================================

def run_unified_vehicle_ai(

    image: Image.Image,
    enable_tyre_analysis: bool = True

) -> Dict[str, Any]:


    # ========================================================
    # STEP 1
    # VEHICLE PART DETECTION
    # ========================================================

    print()
    print("-" * 60)

    print(
        "STEP 1: VEHICLE PART DETECTION"
    )

    print("-" * 60)


    part_detections = (

        detect_vehicle_parts(
            image
        )
    )


    print(
        "Parts detected:",
        len(part_detections)
    )


    # ========================================================
    # STEP 2
    # FIND TYRE
    # ========================================================

    best_tyre_detection = (

        find_best_tyre_detection(
            part_detections
        )
        if enable_tyre_analysis
        else None
    )


    tyre_prediction = None

    image_type = "vehicle"

    tyre_detection_source = None


    # ========================================================
    # CASE A
    # TYRE DETECTED BY YOLO
    # ========================================================

    if best_tyre_detection:

        print(
            "TYRE DETECTED BY YOLO"
        )


        try:

            tyre_crop = crop_tyre_region(

                image,

                best_tyre_detection
            )


            tyre_prediction = (

                predict_tyre_condition(
                    tyre_crop
                )
            )


            image_type = (
                "vehicle_with_detected_tyre"
            )

            tyre_detection_source = (
                "vehicle_parts_yolo"
            )


        except Exception as error:

            print(
                "TYRE CROP/PREDICTION ERROR:"
            )

            print(
                repr(error)
            )


    # ========================================================
    # CASE B
    # NO PART DETECTED
    #
    # Try tyre model directly as fallback for close-up tyre
    # photographs.
    # ========================================================

    elif enable_tyre_analysis and len(part_detections) == 0:

        print(
            "NO VEHICLE PART DETECTED."
        )

        print(
            "TRYING TYRE CLOSE-UP FALLBACK..."
        )


        try:

            fallback_prediction = (

                predict_tyre_condition(
                    image
                )
            )


            print(
                "Tyre fallback confidence:",
                fallback_prediction[
                    "confidence_percent"
                ]
            )


            # Only accept as a tyre close-up when confidence
            # exceeds the safety threshold.

            if (

                fallback_prediction[
                    "confidence"
                ]

                >=

                TYRE_FALLBACK_CONFIDENCE
            ):

                tyre_prediction = (
                    fallback_prediction
                )

                image_type = (
                    "tyre_closeup"
                )

                tyre_detection_source = (
                    "tyre_model_fallback"
                )

                print(
                    "TYRE CLOSE-UP ACCEPTED"
                )


            else:

                print(
                    "TYRE FALLBACK CONFIDENCE TOO LOW"
                )


        except Exception as error:

            print(
                "TYRE FALLBACK ERROR:"
            )

            print(
                repr(error)
            )


    # ========================================================
    # CASE C
    # VEHICLE PARTS DETECTED BUT NO TYRE
    # ========================================================

    else:

        image_type = (
            "vehicle_part"
        )


    # ========================================================
    # STEP 3
    # DAMAGE DETECTION
    #
    # We still run damage detection, but only detections above
    # the configured confidence threshold are returned.
    # ========================================================

    print()
    print("-" * 60)

    print(
        "STEP 3: DAMAGE DETECTION"
    )

    print("-" * 60)


    damage_detections = (

        detect_vehicle_damage(
            image
        )
    )


    print(
        "Damage detections:",
        len(damage_detections)
    )


    # ========================================================
    # STEP 4
    # SELECT BEST PART
    # ========================================================

    best_part = (

        part_detections[0]

        if part_detections

        else None
    )


    # ========================================================
    # STEP 5
    # SELECT BEST DAMAGE
    # ========================================================

    best_damage = (

        damage_detections[0]

        if damage_detections

        else None
    )


    # ========================================================
    # IF TYRE IS DETECTED
    # ========================================================

    tyre_health = None

    if tyre_prediction:

        tyre_condition = tyre_prediction["condition"]
        tyre_confidence = tyre_prediction["confidence"]

        tyre_health = get_tyre_health(
            tyre_condition,
            tyre_confidence
        )

        combined_diagnosis = (
            f"Tyre tread condition detected: {tyre_condition}"
        )

        combined_confidence = tyre_confidence

        tyre_health_score = tyre_health["tyre_health_score"]
        tyre_status = tyre_health["condition_level"]

        requires_user_details = False

        suggested_questions = []


    # ========================================================
    # VEHICLE PART + DAMAGE
    # ========================================================

    elif best_part and best_damage:

        combined_diagnosis = (

            f"{best_damage['class']} detected "
            f"on {best_part['class']}"
        )


        combined_confidence = (

            best_part[
                "confidence"
            ]

            +

            best_damage[
                "confidence"
            ]

        ) / 2


        tyre_health_score = None

        tyre_status = None

        requires_user_details = False

        suggested_questions = []


    # ========================================================
    # FULL VEHICLE / PART IMAGE
    #
    # Ask user for additional context.
    # ========================================================

    elif best_part:

        combined_diagnosis = (

            f"{best_part['class']} detected. "
            "Additional information may help provide "
            "a more detailed diagnosis."
        )


        combined_confidence = (

            best_part[
                "confidence"
            ]
        )


        tyre_health_score = None

        tyre_status = None

        requires_user_details = True


        suggested_questions = [

            "Which vehicle component are you concerned about?",

            "Have you noticed any unusual sound or vibration?",

            "When did you first notice the problem?"
        ]


    # ========================================================
    # NOTHING CONFIDENTLY DETECTED
    # ========================================================

    else:

        combined_diagnosis = (

            "Unable to confidently identify the "
            "vehicle component."
        )


        combined_confidence = 0.0

        tyre_health_score = None

        tyre_status = None

        requires_user_details = True


        suggested_questions = [

            "Which vehicle part would you like to inspect?",

            "Please upload a clearer and closer photograph.",

            "Describe any issue you have noticed while driving."
        ]


    # ========================================================
    # CONFIDENCE LEVEL
    # ========================================================

    confidence_level = (

        get_confidence_level(
            combined_confidence
        )
    )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if tyre_health:

        recommendation = tyre_health[
            "recommendation"
        ]

    else:

        recommendation = (

            generate_recommendation(

                tyre_prediction,

                best_part,

                best_damage
            )
        )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "success":
            True,


        # ----------------------------------------------------
        # IMAGE ANALYSIS TYPE
        # ----------------------------------------------------

        "image_type":
            image_type,


        # ----------------------------------------------------
        # MODELS USED
        # ----------------------------------------------------

        "models_used": {

            "vehicle_parts_yolo":

                vehicle_part_model
                is not None,


            "damage_yolo":

                damage_model
                is not None,


            "tyre_condition_efficientnet":

                tyre_prediction
                is not None
        },


        # ----------------------------------------------------
        # MAIN MODEL RESULT
        # ----------------------------------------------------

        "model": {

            "part":

                best_part["class"]

                if best_part

                else (

                    "Tyre"

                    if tyre_prediction

                    else None
                ),


            "damage":

                best_damage["class"]

                if best_damage

                else None
        },


        # ----------------------------------------------------
        # VEHICLE PART
        # ----------------------------------------------------

        "affected_part":

            best_part["class"]

            if best_part

            else (

                "Tyre"

                if tyre_prediction

                else None
            ),


        "part_confidence":

            best_part["confidence"]

            if best_part

            else None,


        "part_confidence_percent":

            best_part[
                "confidence_percent"
            ]

            if best_part

            else None,


        # ----------------------------------------------------
        # DAMAGE
        # ----------------------------------------------------

        "detected_condition":

            best_damage["class"]

            if best_damage

            else None,


        "damage_confidence":

            best_damage["confidence"]

            if best_damage

            else None,


        "damage_confidence_percent":

            best_damage[
                "confidence_percent"
            ]

            if best_damage

            else None,


        # ----------------------------------------------------
        # TYRE ANALYSIS
        # ----------------------------------------------------

        "tyre_condition":

            tyre_prediction[
                "condition"
            ]

            if tyre_prediction

            else None,


        "tyre_confidence":

            tyre_prediction[
                "confidence"
            ]

            if tyre_prediction

            else None,


        "tyre_confidence_percent":

            tyre_prediction[
                "confidence_percent"
            ]

            if tyre_prediction

            else None,


        "tyre_detection_source":

            tyre_detection_source,


        "tyre_health_score":

            tyre_health_score,


        "tyre_status":

            tyre_status,


        "tyre_condition_level":

            tyre_health["condition_level"]
            if tyre_health
            else None,


        "tyre_analysis_status":

            tyre_health["status"]
            if tyre_health
            else None,


        "tyre_recommendation":

            tyre_health["recommendation"]
            if tyre_health
            else None,


        "top_tyre_predictions":

            tyre_prediction[
                "top_predictions"
            ]

            if tyre_prediction

            else [],


        # ----------------------------------------------------
        # COMBINED DIAGNOSIS
        # ----------------------------------------------------

        "combined_diagnosis":

            combined_diagnosis,


        "combined_confidence":

            round(
                combined_confidence,
                6
            ),


        "combined_confidence_percent":

            round(
                combined_confidence * 100,
                2
            ),


        "confidence_level":

            confidence_level,


        "recommendation":

            recommendation,


        # ----------------------------------------------------
        # USER INPUT REQUIREMENT
        # ----------------------------------------------------

        "requires_user_details":

            requires_user_details,


        "suggested_questions":

            suggested_questions,


        # ----------------------------------------------------
        # ALL DETECTIONS
        # ----------------------------------------------------

        "top_parts":

            part_detections,


        "top_damages":

            damage_detections
    }


# ============================================================
# MAIN PREDICTION ENDPOINT
#
# This matches your existing mobile application:
#
# /predict/vehicle
# ============================================================

@router.post(
    "/predict/vehicle"
)

async def predict_vehicle(

    file: UploadFile = File(...),

    inspection_type: str = Form("auto")

) -> Dict[str, Any]:


    try:

        inspection_type = (
            inspection_type or "auto"
        ).strip().lower()

        if inspection_type not in {
            "auto",
            "body",
            "tyre"
        }:
            raise HTTPException(
                status_code=400,
                detail=(
                    "inspection_type must be one of: "
                    "auto, body, tyre"
                )
            )

        print()
        print("=" * 70)

        print(
            "RIDEGUARDIAN VEHICLE AI REQUEST"
        )

        print("=" * 70)


        # ====================================================
        # VALIDATE CONTENT TYPE
        # ====================================================

        if not file.content_type:

            raise HTTPException(

                status_code=400,

                detail=(
                    "Unable to determine "
                    "uploaded image type."
                )
            )


        if not file.content_type.startswith(
            "image/"
        ):

            raise HTTPException(

                status_code=400,

                detail=(
                    "Uploaded file must be an image."
                )
            )


        # ====================================================
        # READ IMAGE
        # ====================================================

        image_bytes = await file.read()


        if not image_bytes:

            raise HTTPException(

                status_code=400,

                detail=(
                    "Uploaded image is empty."
                )
            )


        # ====================================================
        # OPEN IMAGE
        # ====================================================

        try:

            image = Image.open(

                io.BytesIO(
                    image_bytes
                )

            ).convert(
                "RGB"
            )


        except Exception as error:

            raise HTTPException(

                status_code=400,

                detail=(
                    "Unable to process uploaded "
                    "file as an image."
                )
            ) from error


        print(
            "Image size:",
            image.size
        )


        # ====================================================
        # RUN AI
        #
        # TYRE: use the dedicated EfficientNet-B0 directly.
        # This is the preferred path for a close-up tyre photo
        # because there is currently no dedicated tyre YOLO in
        # the project.
        #
        # BODY: run the existing vehicle-parts + damage YOLO.
        #
        # AUTO: run the existing unified workflow, including the
        # tyre fallback when appropriate.
        # ====================================================

        with inference_lock:

            if inspection_type == "tyre":

                tyre_prediction = predict_tyre_condition(
                    image
                )

                tyre_health = get_tyre_health(
                    tyre_prediction["condition"],
                    tyre_prediction["confidence"]
                )

                result = {
                    "success": True,
                    "inspection_type": "tyre",
                    "image_type": "tyre_closeup",
                    "models_used": {
                        "vehicle_parts_yolo": False,
                        "damage_yolo": False,
                        "tyre_condition_efficientnet": True
                    },
                    "model": {
                        "part": "Tyre",
                        "damage": None
                    },
                    "affected_part": "Tyre",
                    "part_confidence": None,
                    "part_confidence_percent": None,
                    "detected_condition": None,
                    "damage_confidence": None,
                    "damage_confidence_percent": None,
                    "tyre_condition": tyre_prediction["condition"],
                    "tyre_confidence": tyre_prediction["confidence"],
                    "tyre_confidence_percent": tyre_prediction["confidence_percent"],
                    "tyre_detection_source": "dedicated_tyre_condition_model",
                    "tyre_health_score": tyre_health["tyre_health_score"],
                    "tyre_status": tyre_health["condition_level"],
                    "tyre_condition_level": tyre_health["condition_level"],
                    "tyre_analysis_status": tyre_health["status"],
                    "tyre_recommendation": tyre_health["recommendation"],
                    "top_tyre_predictions": tyre_prediction["top_predictions"],
                    "combined_diagnosis": (
                        f"Tyre tread condition detected: "
                        f"{tyre_prediction['condition']}"
                    ),
                    "combined_confidence": tyre_prediction["confidence"],
                    "combined_confidence_percent": tyre_prediction["confidence_percent"],
                    "confidence_level": get_confidence_level(
                        tyre_prediction["confidence"]
                    ),
                    "recommendation": tyre_health["recommendation"],
                    "requires_user_details": False,
                    "suggested_questions": [],
                    "top_parts": [],
                    "top_damages": []
                }

            else:

                result = run_unified_vehicle_ai(
                    image,
                    enable_tyre_analysis=(
                        inspection_type == "auto"
                    )
                )

                result["inspection_type"] = inspection_type


        print()
        print("=" * 70)

        print(
            "FINAL VEHICLE DIAGNOSIS"
        )

        print("=" * 70)

        print(
            "Image Type:",
            result[
                "image_type"
            ]
        )

        print(
            "Diagnosis:",
            result[
                "combined_diagnosis"
            ]
        )

        print(
            "Confidence:",
            result[
                "combined_confidence_percent"
            ]
        )

        print("=" * 70)


        return result


    except HTTPException:

        raise


    except Exception as error:

        print()
        print("=" * 70)

        print(
            "RIDEGUARDIAN VEHICLE AI ERROR"
        )

        print("=" * 70)

        print(
            repr(error)
        )

        print("=" * 70)


        raise HTTPException(

            status_code=500,

            detail=str(
                error
            )
        )


# ============================================================
# DEDICATED TYRE ENDPOINT
#
# Useful for direct mobile/API testing:
# POST /predict/tyre
# ============================================================

@router.post(
    "/predict/tyre"
)
async def predict_tyre(
    file: UploadFile = File(...)
) -> Dict[str, Any]:

    return await predict_vehicle(
        file,
        "tyre"
    )


# ============================================================
# OPTIONAL COMPATIBILITY ENDPOINT
#
# This allows:
#
# POST /vehicle
#
# Useful for your existing Swagger tests.
# ============================================================

@router.post(
    "/vehicle"
)

async def predict_vehicle_legacy(

    file: UploadFile = File(...),

    inspection_type: str = Form("auto")

) -> Dict[str, Any]:

    return await predict_vehicle(
        file,
        inspection_type
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@router.get(
    "/vehicle-ai/health"
)

def vehicle_ai_health():

    return {

        "service":

            "RideGuardian Unified Vehicle AI",


        "status":

            "online",


        "device":

            str(
                DEVICE
            ),


        "models": {

            "vehicle_parts_yolo":

                vehicle_part_model
                is not None,


            "damage_yolo":

                damage_model
                is not None,


            "tyre_condition_efficientnet":

                tyre_model
                is not None
        },


        "tyre_classes":

            TYRE_CLASSES,

        "inspection_types": [
            "auto",
            "body",
            "tyre"
        ]
    }