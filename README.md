# Laser-AI
A chess AI that plays like a pro!

## How to train
1. **Initialize venv**: `python -m venv venv`
2. **Source venv**: `source ./venv/bin/activate`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Extract or generate white_games.csv, white_moves.csv, black_games.csv and black_moves.csv to the "CSV" directory in the repo's root**
   -> [The current dataset](https://drive.google.com/drive/folders/1byzmIGAEhYmlFtMLSG9KKKio5aXHMzCL?usp=sharing)
   
6. **Initialize the model (comments are input in program)**: `python app.py   # 3, 1 or 2, 3, any name (don't use weird chars)`
7. **Train the model**: `python app.py   # 2, 1 (for first run), 0 (if only one model), 3`

## How to use
1. **Initialize venv**: `python -m venv venv`
2. **Source venv**: `source ./venv/bin/activate`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Run the AI locally**:
    * _4.1_ **server**: `fastapi run server.py --port=57664`
    * _4.2_ **client**: `python app.py --ip=127.0.0.1 --port=57664`
