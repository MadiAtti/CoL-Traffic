

def player_specific_metrics(metrics):
    client1_acc = metrics[0][1]["accuracy"] if len(metrics) > 0 else 0
    client2_acc = metrics[1][1]["accuracy"] if len(metrics) > 1 else 0

    return {
        "client1_accuracy": client1_acc,
        "client2_accuracy": client2_acc
    }