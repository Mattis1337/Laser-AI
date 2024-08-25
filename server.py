# REST API
#from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
# AI interaction
import chess
import train_model


app = FastAPI()


class PredictRequest(BaseModel):
    fen: str


class MoveResponse(BaseModel):
    move: str


@app.post("/predict", response_model=MoveResponse)
async def get_prediction_request(request: PredictRequest):
    move = predict_move(fen=request.fen, depth=request.depth)
    return MoveResponse(move)


def predict_move(fen: str, depth: int = 1):
    board = chess.Board(fen)
    ai_moves = train_model.generate_move(color=board.turn, fen=fen, amount_outputs=depth)

    move = None
    for ai_move in ai_moves:
        if ai_move in board.legal_moves:
            move = ai_move
            break
    if not move:
        return predict_move(fen, depth + 1)
         
    return move
