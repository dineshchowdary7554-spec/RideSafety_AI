from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


# ============================================================
# FEDERATED LEARNING API MODELS
# ============================================================

class FLClientRegistration(BaseModel):

    clientId: str = Field(
        ...,
        min_length=3,
        max_length=200
    )

    appVersion: str = "1.0.0"

    platform: str = "unknown"


class FLModelUpdate(BaseModel):

    clientId: str = Field(
        ...,
        min_length=3,
        max_length=200
    )

    roundId: int = Field(
        ...,
        ge=1
    )

    sampleCount: int = Field(
        ...,
        ge=1,
        le=10_000_000
    )

    modelUpdate: Dict[str, float]

    metadata: Optional[Dict[str, Any]] = None


class FLRoundRequest(BaseModel):

    requiredClients: int = Field(
        default=3,
        ge=2,
        le=1000
    )


# ============================================================
# RIDE DATA MODELS
# ============================================================

class RideDataPoint(BaseModel):

    timestamp: Optional[str] = None

    speedKmh: Optional[float] = None

    accelerationX: Optional[float] = None

    accelerationY: Optional[float] = None

    accelerationZ: Optional[float] = None

    gyroX: Optional[float] = None

    gyroY: Optional[float] = None

    gyroZ: Optional[float] = None

    timeSincePreviousReadingSec: Optional[float] = None

    speedChangeKmh: Optional[float] = None

    speedAccelerationMs2: Optional[float] = None

    gyroMagnitudeRadS: Optional[float] = None

    latitude: Optional[float] = None

    longitude: Optional[float] = None

    riskLabel: Optional[int] = Field(
        default=None,
        ge=0,
        le=1
    )

    extra: Optional[Dict[str, Any]] = None


class RideDatasetUpload(BaseModel):

    clientId: str = Field(
        ...,
        min_length=3,
        max_length=200
    )

    rideId: str = Field(
        ...,
        min_length=1,
        max_length=200
    )

    records: List[RideDataPoint] = Field(
        ...,
        min_length=1,
        max_length=50000
    )

    metadata: Optional[Dict[str, Any]] = None


class DatasetExportRequest(BaseModel):

    clientId: str = Field(
        ...,
        min_length=3,
        max_length=200
    )

    includeLabels: bool = True


# ============================================================
# RIDEGUARDIAN FEDERATED LEARNING MODEL
# ============================================================

import numpy as np


class RideRiskModel:

    """
    Lightweight binary logistic regression model
    for RideGuardian Federated Learning.

    Prediction:
        0 = Normal
        1 = Risk Event
    """

    # ========================================================
    # MODEL FEATURES
    # ========================================================

    DEFAULT_FEATURE_NAMES = [

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

        "gyroMagnitudeRadS"

    ]


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self, feature_names=None):

        if feature_names is None:

            feature_names = (
                self.DEFAULT_FEATURE_NAMES.copy()
            )

        self.feature_names = list(
            feature_names
        )

        number_of_features = len(
            self.feature_names
        )


        # Small deterministic initialization.
        #
        # This avoids an entirely symmetric starting
        # state while keeping the model reproducible.

        self.weights = np.full(

            number_of_features,

            0.001,

            dtype=np.float64

        )


        # Slight negative starting bias because
        # risk events are rare.

        self.bias = -0.1


    # ========================================================
    # SIGMOID
    # ========================================================

    @staticmethod
    def sigmoid(z):

        z = np.clip(

            z,

            -50.0,

            50.0

        )


        return (

            1.0

            /

            (

                1.0

                +

                np.exp(

                    -z

                )

            )

        )


    # ========================================================
    # GET FEATURE NAMES
    # ========================================================

    def get_feature_names(self):

        return list(

            self.feature_names

        )


    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    def predict_proba(self, X):

        X = np.asarray(

            X,

            dtype=np.float64

        )


        if X.ndim != 2:

            raise ValueError(

                "X must be a 2-dimensional "
                "feature matrix."

            )


        expected_features = len(

            self.weights

        )


        if X.shape[1] != expected_features:

            raise ValueError(

                f"Expected {expected_features} "
                f"features but received "
                f"{X.shape[1]}."

            )


        linear_output = (

            np.dot(

                X,

                self.weights

            )

            +

            self.bias

        )


        probabilities = self.sigmoid(

            linear_output

        )


        return probabilities


    # ========================================================
    # PREDICT CLASS
    # ========================================================

    def predict(

        self,

        X,

        threshold=0.5

    ):

        probabilities = (

            self.predict_proba(

                X

            )

        )


        predictions = (

            probabilities

            >=

            float(threshold)

        )


        return predictions.astype(

            int

        )


    # ========================================================
    # CALCULATE POSITIVE CLASS WEIGHT
    # ========================================================

    @staticmethod
    def calculate_positive_class_weight(

        y,

        maximum_weight=5.0

    ):

        y = np.asarray(

            y

        ).reshape(

            -1

        )


        positive_samples = int(

            np.sum(

                y == 1

            )

        )


        negative_samples = int(

            np.sum(

                y == 0

            )

        )


        # No positive samples.

        if positive_samples == 0:

            return 1.0


        # No negative samples.

        if negative_samples == 0:

            return 1.0


        # Use square-root weighting instead of
        # the full negative/positive ratio.
        #
        # Example:
        # 1000 normal and 40 risk events
        #
        # Full ratio = 25
        # Square-root ratio = 5

        class_ratio = (

            negative_samples

            /

            positive_samples

        )


        weight = np.sqrt(

            class_ratio

        )


        weight = min(

            float(weight),

            float(maximum_weight)

        )


        weight = max(

            float(weight),

            1.0

        )


        return weight


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    def fit(

        self,

        X,

        y,

        learning_rate=0.01,

        epochs=5,

        positive_class_weight=1.0,

        l2_regularization=0.001,

        gradient_clip_value=5.0

    ):

        X = np.asarray(

            X,

            dtype=np.float64

        )


        y = np.asarray(

            y,

            dtype=np.float64

        ).reshape(

            -1

        )


        if X.ndim != 2:

            raise ValueError(

                "X must be a 2-dimensional "
                "feature matrix."

            )


        if len(X) != len(y):

            raise ValueError(

                "X and y must contain the same "
                "number of samples."

            )


        if len(X) == 0:

            raise ValueError(

                "Training data cannot be empty."

            )


        if X.shape[1] != len(

            self.weights

        ):

            raise ValueError(

                f"Expected {len(self.weights)} "
                f"features but received "
                f"{X.shape[1]}."

            )


        learning_rate = float(

            learning_rate

        )


        positive_class_weight = float(

            positive_class_weight

        )


        positive_class_weight = max(

            positive_class_weight,

            1.0

        )


        sample_count = len(

            X

        )


        # ====================================================
        # TRAINING LOOP
        # ====================================================

        for _ in range(

            int(epochs)

        ):


            # -----------------------------------------------
            # FORWARD PASS
            # -----------------------------------------------

            probabilities = (

                self.predict_proba(

                    X

                )

            )


            # -----------------------------------------------
            # CLASS WEIGHTS
            # -----------------------------------------------

            sample_weights = np.where(

                y == 1,

                positive_class_weight,

                1.0

            )


            # Normalize weights so that the overall
            # gradient magnitude remains stable.

            sample_weights = (

                sample_weights

                /

                np.mean(

                    sample_weights

                )

            )


            # -----------------------------------------------
            # ERROR
            # -----------------------------------------------

            error = (

                probabilities

                -

                y

            )


            weighted_error = (

                error

                *

                sample_weights

            )


            # -----------------------------------------------
            # GRADIENTS
            # -----------------------------------------------

            weight_gradient = (

                np.dot(

                    X.T,

                    weighted_error

                )

                /

                sample_count

            )


            bias_gradient = (

                np.mean(

                    weighted_error

                )

            )


            # -----------------------------------------------
            # L2 REGULARIZATION
            # -----------------------------------------------

            weight_gradient += (

                l2_regularization

                *

                self.weights

            )


            # -----------------------------------------------
            # GRADIENT CLIPPING
            # -----------------------------------------------

            weight_gradient = np.clip(

                weight_gradient,

                -gradient_clip_value,

                gradient_clip_value

            )


            bias_gradient = float(

                np.clip(

                    bias_gradient,

                    -gradient_clip_value,

                    gradient_clip_value

                )

            )


            # -----------------------------------------------
            # UPDATE PARAMETERS
            # -----------------------------------------------

            self.weights -= (

                learning_rate

                *

                weight_gradient

            )


            self.bias -= (

                learning_rate

                *

                bias_gradient

            )


            # -----------------------------------------------
            # NUMERICAL SAFETY
            # -----------------------------------------------

            self.weights = np.nan_to_num(

                self.weights,

                nan=0.0,

                posinf=10.0,

                neginf=-10.0

            )


            self.bias = float(

                np.nan_to_num(

                    self.bias,

                    nan=0.0,

                    posinf=10.0,

                    neginf=-10.0

                )

            )


        return self


    # ========================================================
    # GET PARAMETERS
    # ========================================================

    def get_parameters(self):

        parameters = {}


        for index, feature_name in enumerate(

            self.feature_names

        ):

            parameters[

                feature_name

            ] = float(

                self.weights[index]

            )


        parameters[

            "bias"

        ] = float(

            self.bias

        )


        return parameters


    # ========================================================
    # SET PARAMETERS
    # ========================================================

    def set_parameters(

        self,

        parameters

    ):

        if not isinstance(

            parameters,

            dict

        ):

            raise ValueError(

                "Model parameters must be "
                "provided as a dictionary."

            )


        new_weights = []


        for feature_name in self.feature_names:

            if feature_name not in parameters:

                raise ValueError(

                    f"Missing model parameter: "
                    f"{feature_name}"

                )


            new_weights.append(

                float(

                    parameters[

                        feature_name

                    ]

                )

            )


        if "bias" not in parameters:

            raise ValueError(

                "Missing model parameter: bias"

            )


        self.weights = np.asarray(

            new_weights,

            dtype=np.float64

        )


        self.bias = float(

            parameters[

                "bias"

            ]

        )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def get_model_info(self):

        return {

            "modelType":

                "Binary Logistic Regression",

            "numberOfFeatures":

                int(

                    len(

                        self.feature_names

                    )

                ),

            "featureNames":

                self.get_feature_names(),

            "numberOfParameters":

                int(

                    len(

                        self.feature_names

                    )

                    +

                    1

                )

        }