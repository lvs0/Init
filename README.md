# Init — Intelligence Artificielle Optimisée

**Init** est une architecture complète pour faire tourner des modèles LLM massifs (70B+) sur du matériel limité (2GB RAM, CPU/GPU intégré).

Basé sur le Lenovo X250 et optimisé pour une activation intelligente des couches neurales.

## 🎯 Objectif

Faire tourner des super-intelligences sur du matériel grand public, avec:

- **2GB RAM** maximum
- **Activation intelligente** — pas toutes les couches pour chaque requête
- **Temps de réponse instantané** (<500ms pour queries simples)
- **Fenêtre de contexte flexible** sans explosion mémoire

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Init Engine                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────┐ │
│  │ BrainRouter  │────▶│ LayerStreamer│────▶│ Memory  │ │
│  │  (Activation)│     │   (On-Demand) │     │  Guard  │ │
│  └──────────────┘     └──────────────┘     └─────────┘ │
│         │                    │                 │          │
│         ▼                    ▼                 ▼          │
│  ┌──────────────────────────────────────────────────┐ │
│  │              ContextCompressor                      │ │
│  │         (Fenêtre de contexte large)               │ │
│  └──────────────────────────────────────────────────┘ │
│                          │                              │
│                          ▼                              │
│  ┌──────────────────────────────────────────────────┐ │
│  │              OpenAI-Compatible API               │ │
│  │              /v1/chat/completions                 │ │
│  └──────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🔬 Innovation Principale

### BrainRouter — Activation Différenciée

Pas toutes les requêtes ont besoin du modèle complet.

| Complexité | Activation | Exemples |
|------------|------------|----------|
| Triviale | 10% | "Salut", "Merci" |
| Simple | 25% | "C'est quoi X?" |
| Modérée | 40% | "Explique le concept de Y" |
| Complexe | 60% | "Compare A et B" |
| Avancée | 80% | "Analyse les implications de X" |
| Expert | 100% | "Prouve le théorème de X" |

### LayerStreamer — Chargement à la Demande

- Charge uniquement les couches nécessaires depuis le disque
- Utilise `mmap` pour accès mémoire efficient
- Éjecte les couches les moins utilisées sous pression mémoire

### MemoryGuard — Protection Mémoire

- Surveille RAM/VRAM en temps réel
- Éjecte le KV cache avant OOM
- Garde toujours 10% de buffer

### ContextCompressor — Contexte Long

- Compresse les tokens anciens pour les grandes fenêtres
- Garde sémantique des premiers et derniers tokens
- Échantillonne le milieu intelligemment

## 📦 Installation

```bash
# Cloner le projet
git clone https://github.com/lvs0/Init
cd Init

# Installer dépendances
pip install -r requirements.txt

# Télécharger modèle (exemple Llama-70B)
# (Instructions à venir)

# Lancer
python -m init.src.engine --model-path ./models/llama-70b --max-ram 1.8
```

## 🚀 Utilisation

### API OpenAI-Compatible

```python
import openai

client = openai.OpenAI(
    api_key="none",
    base_url="http://localhost:8080/v1"
)

response = client.chat.completions.create(
    model="llama-70b",
    messages=[
        {"role": "user", "content": "Explique la relativité"}
    ],
    max_tokens=512,
    temperature=0.7
)
```

### CLI

```bash
# Test simple
python -m init.src.engine --model-path ./models/llama-70b --verbose

# Avec API
python -m init.src.engine --model-path ./models/llama-70b --port 8080
```

### Python API

```python
from init.src.engine import InitEngine, ModelConfig, InferenceRequest

config = ModelConfig(
    name="llama-70b",
    path="./models/llama-70b",
    total_layers=80,
    hidden_size=4096,
    num_heads=32,
    vocab_size=32000,
    max_context=8192,
    recommended_ram=4,
    recommended_vram=6,
)

engine = InitEngine(config, max_ram_gb=1.8)
engine.initialize()

request = InferenceRequest(
    prompt="Quel est le sens de la vie?",
    max_tokens=256,
    temperature=0.8
)

response = await engine.infer(request)
print(response.text)
```

## 🔧 Configuration

### Variables d'Environnement

| Variable | Description | Défaut |
|----------|------------|--------|
| `INIT_MAX_RAM` | RAM maximum (GB) | 1.8 |
| `INIT_MAX_VRAM` | VRAM maximum (GB) | 0 |
| `INIT_LOG_LEVEL` | Niveau de log | INFO |

### Options CLI

```
--model PATH           # Nom du modèle
--model-path PATH      # Chemin vers les poids
--layers N             # Nombre de couches (défaut: 80)
--max-ram GB           # RAM maximum (défaut: 1.8)
--max-vram GB           # VRAM maximum (défaut: 0)
--port PORT            # Port API (défaut: 8080)
--verbose              # Mode verbose
```

## 📋 État du Projet

- [x] BrainRouter (analyse + activation)
- [x] LayerStreamer (chargement à la demande)
- [x] MemoryGuard (protection mémoire)
- [x] ContextCompressor (compression contexte)
- [x] InitEngine (orchestrateur)
- [x] API OpenAI-compatible
- [ ] Intégration modèle réel (Llama 70B)
- [ ] Optimisation CUDA/CPU
- [ ] Tests de performance

## 🎓 Licence

MIT — Projet développé par Lévy pour le Polygone Ecosystem

## 👤 Auteur

**Lévy Verpoort Scherpereel** — [polygone@proton.me](mailto:polygone@proton.me)

---

*Init — L'intelligence n'a pas besoin de数据中心.*
