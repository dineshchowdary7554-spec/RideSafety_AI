import numpy as np


def federated_average(

    client_parameters,

    client_sample_counts

):

    """
    Perform sample-weighted Federated Averaging.

    Each rider contributes to the global model
    according to the amount of training data
    used by that rider.
    """


    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if len(

        client_parameters

    ) == 0:

        raise ValueError(

            "No client parameters were provided "
            "for Federated Averaging."

        )


    if len(

        client_parameters

    ) != len(

        client_sample_counts

    ):

        raise ValueError(

            "The number of client models and "
            "sample counts must be equal."

        )


    # ========================================================
    # CONVERT SAMPLE COUNTS
    # ========================================================

    sample_counts = np.asarray(

        client_sample_counts,

        dtype=np.float64

    )


    if np.any(

        sample_counts <= 0

    ):

        raise ValueError(

            "All client sample counts must be "
            "greater than zero."

        )


    total_samples = float(

        np.sum(

            sample_counts

        )

    )


    # ========================================================
    # GET PARAMETER NAMES
    # ========================================================

    parameter_names = list(

        client_parameters[0].keys()

    )


    # ========================================================
    # VALIDATE ALL CLIENT MODELS
    # ========================================================

    for client_index, parameters in enumerate(

        client_parameters

    ):


        if set(

            parameters.keys()

        ) != set(

            parameter_names

        ):

            raise ValueError(

                f"Client {client_index} has "
                "different model parameters."

            )


    # ========================================================
    # FEDERATED AVERAGING
    # ========================================================

    global_parameters = {}


    for parameter_name in parameter_names:


        weighted_sum = 0.0


        for parameters, sample_count in zip(

            client_parameters,

            sample_counts

        ):


            parameter_value = float(

                parameters[

                    parameter_name

                ]

            )


            weighted_sum += (

                parameter_value

                *

                sample_count

            )


        global_parameters[

            parameter_name

        ] = float(

            weighted_sum

            /

            total_samples

        )


    # ========================================================
    # VALIDATE RESULT
    # ========================================================

    for parameter_name, parameter_value in (

        global_parameters.items()

    ):


        if not np.isfinite(

            parameter_value

        ):

            raise ValueError(

                f"FedAvg produced an invalid "
                f"value for {parameter_name}."

            )


    # ========================================================
    # RETURN GLOBAL PARAMETERS
    # ========================================================

    return global_parameters