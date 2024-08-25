# REST API
#from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
# Deps for async AI usage
import asyncio
from concurrent.futures import ThreadPoolExecutor
# AI interaction
import chess
import train_model


executor = ThreadPoolExecutor()
app = FastAPI()


class PredictRequest(BaseModel):
    fen: str


class MoveResponse(BaseModel):
    move: str


@app.post("/predict", response_model=MoveResponse)
async def get_prediction_request(request: PredictRequest):
    move = await asyncio.get_event_loop().run_in_executor(executor, predict_move, request.fen)
    return MoveResponse(move)


def predict_move(fen: str, depth: int = 1):
    board = chess.Board(fen)
    ai_moves = train_model.generate_move(color=board.turn, fen=fen, amount_outputs=depth)

    for ai_move in ai_moves:
        if ai_move in board.legal_moves:
            return ai_move

    return predict_move(fen, depth + 1)
