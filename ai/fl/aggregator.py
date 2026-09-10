from typing import Dict, List


def federated_average(
    client_updates: List[Dict[str, float]],
    client_sample_counts: List[int]
) -> Dict[str, float]:
    """
    Perform Federated Averaging (FedAvg).

    Each client's model parameters are weighted according
    to the number of training samples used by that client.

    Formula:

        Global Model =
        Σ(Client Parameters × Client Sample Count)
        ------------------------------------------
                 Total Sample Count
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if not client_updates:

        raise ValueError(
            "No client model updates were provided."
        )


    if not client_sample_counts:

        raise ValueError(
            "No client sample counts were provided."
        )


    if len(client_updates) != len(
        client_sample_counts
    ):

        raise ValueError(
            "The number of client updates must match "
            "the number of client sample counts."
        )


    # ========================================================
    # VALIDATE SAMPLE COUNTS
    # ========================================================

    for sample_count in client_sample_counts:

        if sample_count <= 0:

            raise ValueError(
                "Every client must have at least "
                "one training sample."
            )


    total_samples = sum(
        client_sample_counts
    )


    if total_samples <= 0:

        raise ValueError(
            "Total sample count must be greater "
            "than zero."
        )


    # ========================================================
    # GET EXPECTED PARAMETERS
    # ========================================================

    first_client_parameters = (
        client_updates[0]
    )

    parameter_names = set(
        first_client_parameters.keys()
    )


    if not parameter_names:

        raise ValueError(
            "Client model parameters are empty."
        )


    # ========================================================
    # VALIDATE ALL CLIENT MODELS
    # ========================================================

    for client_index, client_parameters in enumerate(
        client_updates,
        start=1
    ):

        current_parameter_names = set(
            client_parameters.keys()
        )


        if current_parameter_names != parameter_names:

            missing_parameters = (
                parameter_names
                - current_parameter_names
            )

            extra_parameters = (
                current_parameter_names
                - parameter_names
            )


            raise ValueError(
                f"Client {client_index} has incompatible "
                f"model parameters.\n"
                f"Missing: {sorted(missing_parameters)}\n"
                f"Extra: {sorted(extra_parameters)}"
            )


    # ========================================================
    # FEDERATED AVERAGING
    # ========================================================

    global_parameters = {}


    for parameter_name in sorted(
        parameter_names
    ):

        weighted_sum = 0.0


        for client_parameters, sample_count in zip(

            client_updates,

            client_sample_counts

        ):

            parameter_value = float(

                client_parameters[
                    parameter_name
                ]

            )


            weighted_sum += (

                parameter_value
                * sample_count

            )


        global_parameters[
            parameter_name
        ] = (

            weighted_sum
            / total_samples

        )


    # ========================================================
    # RETURN GLOBAL MODEL PARAMETERS
    # ========================================================

    return global_parameters