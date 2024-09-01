# Laser-AI
A chess AI that plays like a pro!

## How to use
1. **Initialize venv**: `python -m venv venv`
2. **Source venv**: `source ./venv/bin/activate`
3. **Install dependencies**: `pip install numpy python-chess torch pandas requests "fastapi[standard]"`
4. **Run the AI locally**:
    * _4.1_ **server**: `fastapi run server.py --port=57664`
    * _4.2_ **client**: `python app.py --ip=127.0.0.1 --port=57664`
