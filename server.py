# REST API
#from typing import Union
from fastapi import FastAPI, HTTPException
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
    return { "move": move }


def predict_move(fen: str, depth: int = 1):
    try:
        board = chess.Board(fen)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Invalid FEN code",
        )

    if board.legal_moves.count() == 0:
        raise HTTPException(
            status_code=403,
            detail="No moves possible"
        )

    # TODO(Samuil): Isn't a loop that generates
    # all previous moves again + 1 more move very inefficient
    try:
        ai_moves = train_model.generate_move(color=board.turn, fen=fen, amount_outputs=depth)
    except:
        raise HTTPException(
            status_code=406,
            detail="AI not available"
        )

    for ai_move in ai_moves:
        try:
            move = board.parse_san(ai_move)
        except ValueError:
            continue
        if board.is_legal(move):
            return move

    # Make legal_move generator a list, get the first move and then convert it to SAN
    return board.san(list(board.legal_moves)[0])
