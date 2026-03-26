#!/bin/bash

# A lényeg: exportáljuk a jelenlegi mappát a Python elérési útvonalába
export PYTHONPATH=$PYTHONPATH:.

echo "Szerver indítása..."
python3 federated/server.py & 

sleep 5 

echo "Kliensek indítása..."
# Most már látni fogják az utils mappát!
python3 federated/client.py --player P1 --seed 5 &
python3 federated/client.py --player P2 --seed 5 &

wait