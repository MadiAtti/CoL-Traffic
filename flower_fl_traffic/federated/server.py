import flwr as fl

# Ez a függvény fogja lekezelni a kliensektől érkező statisztikákat (accuracy)
def aggregate_metrics(metrics):
    accuracies = [m["accuracy"] for _, m in metrics]
    print(f"\n>>> SZERVER: 1. kör véget ért. Kliensek pontosságai: {accuracies}")
    return {"avg_accuracy": sum(accuracies) / len(accuracies)}

# Stratégia beállítása
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=aggregate_metrics,
)

print("SZERVER: Indulás, várakozás a kliensekre...")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1), # Csak 1 kör a teszthez
    strategy=strategy,
)