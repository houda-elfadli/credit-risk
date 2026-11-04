# Credit Risk - Qualite et organisation d'un projet IA

Projet du TP *Deploiement des Systemes Intelligents* : prediction du risque de
credit avec un code structure (SOLID), analyse avec Ruff et formatage avec Black.

## Structure

```
credit_project/
|-- data/raw/            # donnees (credit.csv, ignore par Git)
|-- notebooks/           # exploration
|-- models/              # modeles entraines
|-- src/credit_risk/     # code source (data, preprocessing, training,
|                        #   evaluation, prediction, pipeline)
|-- tests/               # tests pytest
|-- scripts/             # generation de donnees d'exemple
|-- main.py
`-- pyproject.toml
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Utilisation

```bash
python scripts/generate_sample_data.py   # si vous n'avez pas credit.csv
python main.py
```

## Qualite

```bash
ruff check .
black --check .
pytest
```

## Principes SOLID appliques

- **SRP** : un module par responsabilite.
- **OCP** : ajouter un modele = ajouter une classe `ModelTrainer`.
- **DIP** : `run_pipeline` depend de l'abstraction `ModelTrainer`.
