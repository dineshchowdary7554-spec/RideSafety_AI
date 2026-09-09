# ============================================================
# RIDEGUARDIAN VEHICLE DIAGNOSIS API
#
# Combines:
# 1. Vehicle Parts YOLO
# 2. Vehicle Damage YOLO
# 3. Tyre Condition EfficientNet-B0
# ============================================================

import io
import sys
from pathlib import Path

import torch
import torch.nn as nn

from fastapi import APIRouter, UploadFile, File, HTTPException

from PIL import Image
from torchvision import models, transforms

from ultralytics import YOLO


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/predict",
    tags=["Vehicle AI"]
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("RIDEGUARDIAN VEHICLE AI INITIALIZATION")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# MODEL PATHS
# ============================================================

VEHICLE_PARTS_MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "vehicle_parts_v2"
    / "vehicle_parts_v2"
    / "weights"
    / "best.pt"
)

DAMAGE_MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "damage"
    / "damage_v1"
    / "weights"
    / "best.pt"
)

TYRE_MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "models"
    / "tyre_condition_model.pth"
)


# ============================================================
# TYRE CLASSES
# ============================================================

TYRE_CLASSES = [
    "3mm",
    "5mm",
    "6.5mm",
    "7mm",
    "8mm",
    "under_3mm"
]

TYRE_IMAGE_SIZE = 224


# ============================================================
# LOAD VEHICLE PARTS YOLO
# ============================================================

vehicle_parts_model = None

try:

    if VEHICLE_PARTS_MODEL_PATH.exists():

        vehicle_parts_model = YOLO(
            str(VEHICLE_PARTS_MODEL_PATH)
        )

        print(
            "Vehicle Parts YOLO loaded successfully"
        )

        print(
            "Classes:",
            vehicle_parts_model.names
        )

    else:

        print(
            "WARNING: Vehicle Parts model not found:"
        )

        print(
            VEHICLE_PARTS_MODEL_PATH
        )

except Exception as error:

    print(
        "Vehicle Parts YOLO loading error:"
    )

    print(error)


# ============================================================
# LOAD DAMAGE YOLO
# ============================================================

damage_model = None

try:

    if DAMAGE_MODEL_PATH.exists():

        damage_model = YOLO(
            str(DAMAGE_MODEL_PATH)
        )

        print(
            "Damage YOLO loaded successfully"
        )

        print(
            "Classes:",
            damage_model.names
        )

    else:

        print(
            "WARNING: Damage model not found:"
        )

        print(
            DAMAGE_MODEL_PATH
        )

except Exception as error:

    print(
        "Damage YOLO loading error:"
    )

    print(error)


# ============================================================
# CREATE TYRE MODEL
# ============================================================

def create_tyre_model(
    num_classes
):

    model = models.efficientnet_b0(
        weights=None
    )

    in_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


# ============================================================
# LOAD TYRE MODEL
# ============================================================

tyre_model = None

try:

    if TYRE_MODEL_PATH.exists():

        print(
            "Loading tyre condition model..."
        )

        checkpoint = torch.load(
            str(TYRE_MODEL_PATH),
            map_location=DEVICE
        )

        tyre_model = create_tyre_model(
            num_classes=len(TYRE_CLASSES)
        )

        tyre_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        tyre_model.to(DEVICE)

        tyre_model.eval()

        print(
            "Tyre Condition Model loaded successfully"
        )

        print(
            "Architecture:",
            checkpoint.get(
                "architecture",
                "Unknown"
            )
        )

        print(
            "Classes:",
            checkpoint.get(
                "class_names",
                TYRE_CLASSES
            )
        )

    else:

        print(
            "WARNING: Tyre model not found:"
        )

        print(
            TYRE_MODEL_PATH
        )

except Exception as error:

    print(
        "Tyre model loading error:"
    )

    print(error)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

tyre_transform = transforms.Compose([

    transforms.Resize(
        (TYRE_IMAGE_SIZE, TYRE_IMAGE_SIZE)
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
# HELPER:
# CONVERT YOLO RESULTS
# ============================================================

def get_yolo_predictions(
    results
):

    predictions = []

    for result in results:

        if result.boxes is None:

            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            class_name = result.names[
                class_id
            ]

            coordinates = box.xyxy[
                0
            ].tolist()

            predictions.append({

                "class": str(
                    class_name
                ),

                "confidence": round(
                    confidence,
                    4
                ),

                "confidence_percent": round(
                    confidence * 100,
                    2
                ),

                "bbox": [

                    round(
                        value,
                        2
                    )

                    for value in coordinates

                ]

            })

    predictions.sort(

        key=lambda item:
        item["confidence"],

        reverse=True

    )

    return predictions


# ============================================================
# HELPER:
# FIND TYRE DETECTION
# ============================================================

def find_tyre(
    predictions
):

    tyre_keywords = [

        "tyre",
        "tire",
        "wheel"

    ]

    for prediction in predictions:

        detected_name = (
            prediction["class"]
            .lower()
        )

        for keyword in tyre_keywords:

            if keyword in detected_name:

                return prediction

    return None


# ============================================================
# CROP TYRE FROM IMAGE
# ============================================================

def crop_tyre_from_image(
    image,
    bbox
):

    x1, y1, x2, y2 = bbox

    width, height = image.size

    x1 = max(
        0,
        int(x1)
    )

    y1 = max(
        0,
        int(y1)
    )

    x2 = min(
        width,
        int(x2)
    )

    y2 = min(
        height,
        int(y2)
    )

    if x2 <= x1 or y2 <= y1:

        return image

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
    image
):

    if tyre_model is None:

        return None

    image_tensor = tyre_transform(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = tyre_model(
            image_tensor
        )

        probabilities = torch.softmax(

            outputs,

            dim=1

        )

        confidence, prediction = torch.max(

            probabilities,

            dim=1

        )

    class_index = prediction.item()

    tyre_condition = (
        TYRE_CLASSES[
            class_index
        ]
    )

    confidence_value = (
        confidence.item()
    )

    return {

        "condition":
        tyre_condition,

        "confidence":
        round(
            confidence_value,
            4
        ),

        "confidence_percent":
        round(
            confidence_value * 100,
            2
        )

    }


# ============================================================
# TYRE HEALTH SCORE
# ============================================================

def calculate_tyre_health(
    condition
):

    tyre_health_scores = {

        "8mm": 100,

        "7mm": 92,

        "6.5mm": 85,

        "5mm": 70,

        "3mm": 40,

        "under_3mm": 20

    }

    return tyre_health_scores.get(
        condition,
        50
    )


# ============================================================
# TYRE RECOMMENDATION
# ============================================================

def get_tyre_recommendation(
    condition
):

    recommendations = {

        "8mm":
        "Tyres appear to be in excellent condition.",

        "7mm":
        "Tyres are in very good condition. Continue regular inspections.",

        "6.5mm":
        "Tyres are in good condition with normal wear.",

        "5mm":
        "Moderate tyre wear detected. Monitor tyre condition regularly.",

        "3mm":
        "Significant tyre wear detected. Consider a professional inspection soon.",

        "under_3mm":
        "Critical tyre wear detected. Tyre replacement is strongly recommended."

    }

    return recommendations.get(

        condition,

        "Unable to determine tyre recommendation."

    )


# ============================================================
# MAIN VEHICLE DIAGNOSIS ENDPOINT
# ============================================================

@router.post(
    "/vehicle"
)

async def predict_vehicle(
    file: UploadFile = File(...)
):

    # ========================================================
    # VALIDATE FILE
    # ========================================================

    if not file.content_type:

        raise HTTPException(

            status_code=400,

            detail=
            "Invalid file type."

        )

    if not file.content_type.startswith(
        "image/"
    ):

        raise HTTPException(

            status_code=400,

            detail=
            "Please upload an image file."

        )


    # ========================================================
    # READ IMAGE
    # ========================================================

    try:

        image_bytes = await file.read()

        image = Image.open(

            io.BytesIO(
                image_bytes
            )

        ).convert(
            "RGB"
        )

    except Exception:

        raise HTTPException(

            status_code=400,

            detail=
            "Unable to read the uploaded image."

        )


    # ========================================================
    # RESULT STRUCTURE
    # ========================================================

    response = {

        "success": True,

        "image_type":
        "vehicle",

        "vehicle_parts": [],

        "damage_predictions": [],

        "tyre_analysis": None,

        "combined_diagnosis": None,

        "recommendation": None

    }


    # ========================================================
    # VEHICLE PART DETECTION
    # ========================================================

    vehicle_predictions = []

    if vehicle_parts_model is not None:

        try:

            results = vehicle_parts_model(

                image,

                conf=0.25,

                verbose=False

            )

            vehicle_predictions = (
                get_yolo_predictions(
                    results
                )
            )

            response[
                "vehicle_parts"
            ] = vehicle_predictions

        except Exception as error:

            print(
                "Vehicle detection error:",
                error
            )


    # ========================================================
    # DAMAGE DETECTION
    # ========================================================

    damage_predictions = []

    if damage_model is not None:

        try:

            results = damage_model(

                image,

                conf=0.25,

                verbose=False

            )

            damage_predictions = (
                get_yolo_predictions(
                    results
                )
            )

            response[
                "damage_predictions"
            ] = damage_predictions

        except Exception as error:

            print(
                "Damage detection error:",
                error
            )


    # ========================================================
    # CHECK FOR TYRE
    # ========================================================

    detected_tyre = find_tyre(

        vehicle_predictions

    )


    # ========================================================
    # CASE 1:
    # TYRE DETECTED BY YOLO
    # ========================================================

    if detected_tyre is not None:

        tyre_image = crop_tyre_from_image(

            image,

            detected_tyre[
                "bbox"
            ]

        )

        tyre_result = (
            predict_tyre_condition(
                tyre_image
            )
        )

        if tyre_result:

            condition = tyre_result[
                "condition"
            ]

            tyre_health = (
                calculate_tyre_health(
                    condition
                )
            )

            recommendation = (
                get_tyre_recommendation(
                    condition
                )
            )

            response[
                "image_type"
            ] = "vehicle_with_tyre"

            response[
                "tyre_analysis"
            ] = {

                "detected": True,

                "detected_part":
                detected_tyre[
                    "class"
                ],

                "detection_confidence":
                detected_tyre[
                    "confidence_percent"
                ],

                "condition":
                condition,

                "condition_confidence":
                tyre_result[
                    "confidence_percent"
                ],

                "health_score":
                tyre_health,

                "recommendation":
                recommendation

            }

            response[
                "combined_diagnosis"
            ] = (

                f"Tyre detected and analysed. "
                f"Estimated tyre condition: "
                f"{condition}."

            )

            response[
                "recommendation"
            ] = recommendation


    # ========================================================
    # CASE 2:
    # NO TYRE DETECTED
    # ========================================================

    else:

        response[
            "combined_diagnosis"
        ] = (

            "Vehicle image analysed. "
            "No tyre was confidently detected "
            "for tread condition analysis."

        )

        response[
            "recommendation"
        ] = (

            "For tyre condition analysis, "
            "upload a clear close-up image "
            "of the tyre tread."

        )


    # ========================================================
    # DAMAGE SUMMARY
    # ========================================================

    if len(
        damage_predictions
    ) > 0:

        top_damage = (
            damage_predictions[0]
        )

        response[
            "detected_condition"
        ] = top_damage[
            "class"
        ]

        response[
            "damage_confidence"
        ] = top_damage[
            "confidence"
        ]

        response[
            "damage_confidence_percent"
        ] = top_damage[
            "confidence_percent"
        ]


    # ========================================================
    # TOP PART
    # ========================================================

    if len(
        vehicle_predictions
    ) > 0:

        top_part = (
            vehicle_predictions[0]
        )

        response[
            "affected_part"
        ] = top_part[
            "class"
        ]

        response[
            "part_confidence"
        ] = top_part[
            "confidence"
        ]

        response[
            "part_confidence_percent"
        ] = top_part[
            "confidence_percent"
        ]


    return response