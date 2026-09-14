def player_specific_metrics(metrics):
    '''
    Custom metric aggregation function for Flower that extracts and returns
    the accuracy and loss for each client separately from the evaluation results.
    '''
    client1_metrics = metrics[0][1] if len(metrics) > 0 else {}
    client2_metrics = metrics[1][1] if len(metrics) > 1 else {}

    return {
        "client1_accuracy": client1_metrics.get("accuracy", 0),
        "client2_accuracy": client2_metrics.get("accuracy", 0),
        "client1_loss": client1_metrics.get("loss", 0),
        "client2_loss": client2_metrics.get("loss", 0)
    }