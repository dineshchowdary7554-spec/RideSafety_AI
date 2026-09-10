import os
import pickle
import numpy as np

from fl.models import RideRiskModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "federated",
    "rideguardian_federated_risk_model.pkl"
)


SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "federated",
    "feature_scaler.pkl"
)


# ============================================================
# DEFAULT FEATURES
# ============================================================

FEATURE_NAMES = [

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


# ============================================================
# PRODUCTION THRESHOLD
# ============================================================

DEFAULT_THRESHOLD = 0.90


# ============================================================
# RIDEGUARDIAN FEDERATED PREDICTOR
# ============================================================

class FederatedRiskPredictor:

    def __init__(
        self,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        threshold=DEFAULT_THRESHOLD
    ):

        self.model_path = model_path

        self.scaler_path = scaler_path

        self.threshold = threshold

        self.model = None

        self.scaler = None

        self.load_model()

        self.load_scaler()


    # ========================================================
    # LOAD FEDERATED MODEL
    # ========================================================

    def load_model(self):

        print(
            "Loading federated risk model..."
        )

        if not os.path.exists(
            self.model_path
        ):

            raise FileNotFoundError(

                "Federated model not found:\n"
                f"{self.model_path}"

            )


        with open(
            self.model_path,
            "rb"
        ) as file:

            loaded_object = pickle.load(
                file
            )


        # ----------------------------------------------------
        # CASE 1:
        # Entire RideRiskModel object was saved
        # ----------------------------------------------------

        if isinstance(
            loaded_object,
            RideRiskModel
        ):

            self.model = loaded_object


        # ----------------------------------------------------
        # CASE 2:
        # Only model parameters were saved
        # ----------------------------------------------------

        elif isinstance(
            loaded_object,
            dict
        ):

            self.model = RideRiskModel()

            self.model.set_parameters(
                loaded_object
            )


        else:

            raise ValueError(

                "Unsupported federated model format."

            )


        print(
            "Federated risk model loaded successfully."
        )


    # ========================================================
    # LOAD FEATURE SCALER
    # ========================================================

    def load_scaler(self):

        print(
            "Loading feature scaler..."
        )

        if not os.path.exists(
            self.scaler_path
        ):

            raise FileNotFoundError(

                "Feature scaler not found:\n"
                f"{self.scaler_path}"

            )


        with open(
            self.scaler_path,
            "rb"
        ) as file:

            self.scaler = pickle.load(
                file
            )


        print(
            "Feature scaler loaded successfully."
        )


    # ========================================================
    # VALIDATE SENSOR DATA
    # ========================================================

    def validate_sensor_data(
        self,
        sensor_data
    ):

        if not isinstance(
            sensor_data,
            dict
        ):

            raise ValueError(

                "Sensor data must be a dictionary."

            )


        missing_features = []


        for feature_name in FEATURE_NAMES:

            if feature_name not in sensor_data:

                missing_features.append(
                    feature_name
                )


        if missing_features:

            raise ValueError(

                "Missing required sensor features:\n"
                f"{missing_features}"

            )


        return True


    # ========================================================
    # CONVERT SENSOR DATA TO FEATURE ARRAY
    # ========================================================

    def prepare_features(
        self,
        sensor_data
    ):

        self.validate_sensor_data(
            sensor_data
        )


        feature_values = []


        for feature_name in FEATURE_NAMES:

            value = sensor_data[
                feature_name
            ]


            if value is None:

                raise ValueError(

                    f"Feature '{feature_name}' "
                    "cannot be None."

                )


            try:

                value = float(
                    value
                )


            except (
                ValueError,
                TypeError
            ):

                raise ValueError(

                    f"Feature '{feature_name}' "
                    "must be numeric."

                )


            if not np.isfinite(
                value
            ):

                raise ValueError(

                    f"Feature '{feature_name}' "
                    "must be finite."

                )


            feature_values.append(
                value
            )


        X = np.array(

            [feature_values],

            dtype=np.float64

        )


        return X


    # ========================================================
    # NORMALIZE FEATURES
    # ========================================================

    def normalize_features(
        self,
        X
    ):

        if self.scaler is None:

            raise RuntimeError(

                "Feature scaler is not loaded."

            )


        X_normalized = self.scaler.transform(
            X
        )


        return X_normalized


    # ========================================================
    # PREDICT RISK PROBABILITY
    # ========================================================

    def predict_probability(
        self,
        sensor_data
    ):

        X = self.prepare_features(
            sensor_data
        )


        X_normalized = (
            self.normalize_features(
                X
            )
        )


        probability = (

            self.model.predict_proba(
                X_normalized
            )[0]

        )


        probability = float(
            probability
        )


        return probability


    # ========================================================
    # PREDICT RISK
    # ========================================================

    def predict(
        self,
        sensor_data,
        threshold=None
    ):

        if threshold is None:

            threshold = self.threshold


        threshold = float(
            threshold
        )


        if threshold <= 0 or threshold >= 1:

            raise ValueError(

                "Threshold must be between "
                "0 and 1."

            )


        probability = (

            self.predict_probability(
                sensor_data
            )

        )


        prediction = (

            probability >= threshold

        )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if probability >= 0.90:

            risk_level = "CRITICAL"


        elif probability >= 0.75:

            risk_level = "HIGH"


        elif probability >= 0.50:

            risk_level = "MEDIUM"


        else:

            risk_level = "LOW"


        # ----------------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------------

        result = {

            "prediction":

                "RISK"

                if prediction

                else

                "NORMAL",


            "isRisk":

                bool(
                    prediction
                ),


            "riskProbability":

                round(
                    probability,
                    6
                ),


            "riskPercentage":

                round(
                    probability * 100,
                    2
                ),


            "riskLevel":

                risk_level,


            "threshold":

                round(
                    threshold,
                    4
                ),


            "featuresUsed":

                FEATURE_NAMES.copy()

        }


        return result


    # ========================================================
    # CHANGE DEFAULT THRESHOLD
    # ========================================================

    def set_threshold(
        self,
        threshold
    ):

        threshold = float(
            threshold
        )


        if threshold <= 0 or threshold >= 1:

            raise ValueError(

                "Threshold must be between "
                "0 and 1."

            )


        self.threshold = threshold


        return self.threshold


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    def get_model_info(self):

        return {

            "modelPath":

                self.model_path,


            "scalerPath":

                self.scaler_path,


            "defaultThreshold":

                self.threshold,


            "featureCount":

                len(
                    FEATURE_NAMES
                ),


            "features":

                FEATURE_NAMES.copy(),


            "modelInfo":

                self.model.get_model_info()

        }


# ============================================================
# GLOBAL PREDICTOR INSTANCE
# ============================================================

_predictor_instance = None


def get_federated_predictor():

    global _predictor_instance


    if _predictor_instance is None:

        _predictor_instance = (

            FederatedRiskPredictor()

        )


    return _predictor_instance


# ============================================================
# SIMPLE PREDICTION FUNCTION
# ============================================================

def predict_ride_risk(
    sensor_data,
    threshold=None
):

    predictor = (
        get_federated_predictor()
    )


    return predictor.predict(

        sensor_data,

        threshold=threshold

    )