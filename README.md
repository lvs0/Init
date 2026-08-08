# Init

**Faire tourner des modèles LLM massifs (70B+) sur du matériel limité.**

Init est une architecture d'activation intelligente des couches neuronales, pensée
pour du matériel grand public (testée sur un Lenovo X250 : 2 Go RAM, CPU/GPU intégré).

L'idée : n'activer que les couches nécessaires à chaque token, au lieu de charger
l'ensemble du réseau — rendre la grande échelle possible sans le gros matériel.

## État

- Recherche / prototype (Python)

## Licence

MIT
