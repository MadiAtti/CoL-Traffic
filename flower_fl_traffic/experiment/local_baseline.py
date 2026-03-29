import torch
import torch.nn as nn
import json
import os
from omegaconf import OmegaConf
from models.neural_network import TrafficNN
from utils.evaluation import evaluate_model

def run_local_experiment(config, train_loaders, test_loaders, subdir, client_ids=None):
    """
    Lefuttatja a lokális baseline tanítást (M1 és M2) és a kért JSON formátumban ment.
    """
    # 1. Konfiguráció letisztítása (Hydra -> Sima Python Dict)
    # Ez küszöböli ki a ContainerMetadata hibát
    clean_params = OmegaConf.to_container(config, resolve=True)

    # 2. Alapstruktúra létrehozása (a 0.json minta alapján)
    results = {
        "parameters": clean_params,
        "models": {}
    }

    # Kliensnevek kezelése (ha lista érkezik a main-ből)
    if client_ids is None:
        client_ids = ["P1", "P2"]

    # 3. Kliensek tanítása (M1, M2...)
    for i, train_loader in enumerate(train_loaders):
        model_key = f"M{i+1}"
        print(f"  --- {model_key} lokális tanítása indult ---")
        
        model = TrafficNN(input_dim=config.dataset.input_dim, num_classes=config.dataset.num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.config.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Összes epoch: körök * belső epochok
        total_epochs = config.config.federated_rounds * config.config.num_epochs 
        
        training_history = []
        for epoch in range(1, total_epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
            
            # Epoch szintű statisztika mentése a listába
            training_history.append({
                "epoch": epoch,
                "loss": running_loss / total,
                "accuracy": correct / total
            })
            
            if epoch % 10 == 0:
                print(f"    Epoch {epoch}/{total_epochs} kész.")

        # 4. Értékelés (Cross-evaluation)
        # Itt minden modell lefut minden kliens tesztadatán
        evaluation_results = {}
        for j, test_loader in enumerate(test_loaders):
            target_name = client_ids[j]
            loss, acc = evaluate_model(model, test_loader, criterion)
            evaluation_results[target_name] = {
                "loss": float(loss),
                "accuracy": float(acc)
            }

        # Adatok hozzáadása a fő objektumhoz
        results["models"][model_key] = {
            "training": training_history,
            "evaluation": evaluation_results
        }

    # 5. Mentés (a seed alapján elnevezett fájlba)
    target_path = os.path.join("1_local_baseline", subdir)
    os.makedirs(target_path, exist_ok=True)
    file_path = os.path.join(target_path, f"{config.config.seed}.json")

    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Lokális baseline sikeresen mentve: {file_path}")