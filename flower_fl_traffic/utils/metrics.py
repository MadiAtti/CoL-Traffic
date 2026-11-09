"""Aggregate Flower evaluation metrics by stable traffic-client identity."""


def player_specific_metrics(metrics):
    """Return client-specific accuracy, loss, and epsilon aggregation values.

    Args:
        metrics: Flower's weighted ``(num_examples, metrics)`` client results.

    Returns:
        A dictionary containing one value for each metric and client ID.

    Raises:
        ValueError: If a client ID is missing, duplicated, unexpected, or
            lacks a required metric.
    """
    by_client = _metrics_by_client(metrics)

    return {
        "client1_accuracy": by_client["0"]["accuracy"],
        "client2_accuracy": by_client["1"]["accuracy"],
        "client1_loss": by_client["0"]["loss"],
        "client2_loss": by_client["1"]["loss"],
        "client1_epsilon": by_client["0"]["epsilon"],
        "client2_epsilon": by_client["1"]["epsilon"],
    }


def _metrics_by_client(metrics):
    """Validate and index one evaluation result per configured client."""
    by_client = {}
    for _, values in metrics:
        client_id = values.get("client_id")
        if client_id not in {"0", "1"}:
            raise ValueError(f"Unexpected or missing client_id in metrics: {client_id!r}")
        if client_id in by_client:
            raise ValueError(f"Duplicate metrics for client {client_id}.")
        by_client[client_id] = values

    missing = {"0", "1"} - by_client.keys()
    if missing:
        raise ValueError(f"Missing metrics for client(s): {sorted(missing)}")
    if any(
        "accuracy" not in values
        or "loss" not in values
        or "epsilon" not in values
        for values in by_client.values()
    ):
        raise ValueError(
            "Evaluation metrics must include accuracy, loss, and epsilon for both clients."
        )
    return by_client