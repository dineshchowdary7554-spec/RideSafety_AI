import copy
import numpy as np

from fl.models import RideRiskModel


def train_local_model(

    global_parameters,

    X,

    y,

    epochs=5,

    learning_rate=0.01

):

    """
    Train one local RideRiskModel using
    a rider's private dataset.

    Steps:

    1. Create a fresh local model.
    2. Load global model parameters.
    3. Validate local data.
    4. Analyze class distribution.
    5. Calculate controlled class weight.
    6. Train the local model.
    7. Generate diagnostics.
    8. Return model parameters.
    """


    # ========================================================
    # CREATE LOCAL MODEL
    # ========================================================

    model = RideRiskModel()


    # ========================================================
    # LOAD GLOBAL PARAMETERS
    # ========================================================

    model.set_parameters(

        copy.deepcopy(

            global_parameters

        )

    )


    # ========================================================
    # STORE PARAMETERS BEFORE TRAINING
    # ========================================================

    parameters_before = (

        copy.deepcopy(

            model.get_parameters()

        )

    )


    # ========================================================
    # CONVERT DATA
    # ========================================================

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


    # ========================================================
    # VALIDATE DATA
    # ========================================================

    if len(X) == 0:

        raise ValueError(

            "Local training dataset is empty."

        )


    if len(X) != len(y):

        raise ValueError(

            "X and y must contain the same "
            "number of samples."

        )


    if X.ndim != 2:

        raise ValueError(

            "X must be a 2-dimensional "
            "feature matrix."

        )


    # ========================================================
    # ANALYZE RIDER DATA
    # ========================================================

    total_samples = len(

        y

    )


    risk_samples = int(

        np.sum(

            y == 1

        )

    )


    normal_samples = int(

        np.sum(

            y == 0

        )

    )


    # ========================================================
    # CALCULATE CLASS WEIGHT
    # ========================================================

    calculated_weight = (

        model.calculate_positive_class_weight(

            y,

            maximum_weight=5.0

        )

    )


    # Additional safety limit.

    positive_class_weight = min(

        float(calculated_weight),

        5.0

    )


    positive_class_weight = max(

        float(positive_class_weight),

        1.0

    )


    # ========================================================
    # TRAIN LOCAL MODEL
    # ========================================================

    model.fit(

        X=X,

        y=y,

        learning_rate=learning_rate,

        epochs=epochs,

        positive_class_weight=

            positive_class_weight,

        l2_regularization=0.001,

        gradient_clip_value=5.0

    )


    # ========================================================
    # GET UPDATED PARAMETERS
    # ========================================================

    local_parameters = (

        model.get_parameters()

    )


    # ========================================================
    # PARAMETER CHANGE
    # ========================================================

    parameter_changes = []


    for parameter_name in local_parameters:


        before_value = float(

            parameters_before[

                parameter_name

            ]

        )


        after_value = float(

            local_parameters[

                parameter_name

            ]

        )


        parameter_change = abs(

            after_value

            -

            before_value

        )


        parameter_changes.append(

            parameter_change

        )


    total_parameter_change = float(

        np.sum(

            parameter_changes

        )

    )


    maximum_parameter_change = float(

        np.max(

            parameter_changes

        )

    )


    # ========================================================
    # TRAINING PROBABILITIES
    # ========================================================

    probabilities = (

        model.predict_proba(

            X

        )

    )


    minimum_probability = float(

        np.min(

            probabilities

        )

    )


    maximum_probability = float(

        np.max(

            probabilities

        )

    )


    mean_probability = float(

        np.mean(

            probabilities

        )

    )


    probability_std = float(

        np.std(

            probabilities

        )

    )


    # ========================================================
    # TRAINING INFORMATION
    # ========================================================

    training_info = {


        "totalSamples":

            int(total_samples),


        "normalSamples":

            int(normal_samples),


        "riskSamples":

            int(risk_samples),


        "calculatedPositiveClassWeight":

            float(calculated_weight),


        "usedPositiveClassWeight":

            float(positive_class_weight),


        "epochs":

            int(epochs),


        "learningRate":

            float(learning_rate),


        "totalParameterChange":

            float(total_parameter_change),


        "maximumParameterChange":

            float(maximum_parameter_change),


        "minimumProbability":

            float(minimum_probability),


        "maximumProbability":

            float(maximum_probability),


        "meanProbability":

            float(mean_probability),


        "probabilityStandardDeviation":

            float(probability_std)

    }


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return (

        local_parameters,

        total_samples,

        training_info

    )