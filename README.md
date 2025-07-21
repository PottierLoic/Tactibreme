# Tactibreme

Tactibreme is a tool made to train an AI to play the game `Les tacticiens de Brême` and generate statistics and datasets.

## Features

- Train an AI to play the game
- Play games and record the data
- Generate statistics and datasets (WIP)
- Play against the AI (WIP)

## Requirements

Ensure you have the following installed before running the project:

- Python 3.8+
- Pytorch
- pygame
- matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PottierLoic/tactibreme.git
cd tactibreme
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

TODO section about torchvision if training on GPU

# Usage

### Training a new AI model

```bash
python main.py train model1 --count=10000 # New model
python main.py train model1 --count=500 --load path/to/model1.pth path/to/model2.pth # Continue training existing model
```

### AI vs AI

```bash
python main.py record --blue=path/to/model1.pth --red=path/to/model2.pth --count=1000
```

### Statistics generation (WIP)

```bash
python main.py stats csv=path/to/csv.csv
```

## Complete manual

```
TACTIBREME(1)              User Commands              TACTIBREME(1)

NAME
       tactibreme - CLI tool for training an AI and playing board game matches

SYNOPSIS
       tactibreme [OPTIONS] COMMAND [ARGUMENTS]...

DESCRIPTION
       tactibreme allows training an AI, recording matches, replaying past games,
       and generating statistics from match history.

COMMANDS
       train <name> [count=NB_GAME] [epsilon=VAL] [decay=VAL] [gamma=VAL] [lr=VAL]
              Trains an AI model. Can save match history using `--record`
              and generate statistics with `--stats`.

       record [count=NB_GAME] [blue=MODEL_NAME] [red=MODEL_NAME]
              Plays matches between two models and records the move history.
              Can generate statistics with `--stats`.

       stats csv=FILE.csv
              Generates statistics from a match history file.

OPTIONS
       --record FILE
              Saves detailed move history of matches in a CSV file.
              Works with `train` and `record`. This file can later be used with `demo`.

       --stats FILE
              Automatically generates statistics after `train` or `record`
              and saves them to a file.

       --verbose
              Displays detailed logs during execution.

       --ui
              Enables the graphical interface for live matches.

```


