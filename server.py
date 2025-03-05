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


# parse JSON to objects of Python
class PredictRequest(BaseModel):
    fen: str


class MoveResponse(BaseModel):
    move: str


@app.post("/predict", response_model=MoveResponse)
async def get_prediction_request(request: PredictRequest):
    """
    Accept POST request to domain:port/predict and parse JSON to PredictRequest object

    Args:
        request (PredictRequest): Python object of class BaseModel containing JSON data 

    Returns:
        Dict[str, str]: The move in standardized JSON format defined in MoveResponse
    """
    # run AI on seperate coroutine preventing blockage of event loop
    move = await asyncio.get_event_loop().run_in_executor(executor, predict_move, request.fen)
    return { "move": move } # standardized response 


def predict_move(fen: str, depth: int = 5):
    """
    Predict move by running AI or selecting some legal move as fallback.
    This function parses the FEN provided by the client. If the FEN is valid,
    the AI model of the currently playing color is loaded and moves are requested.
    They are sorted by the "likeliness" of them being the answer to the corresponding
    FEN in the dataset. Because of this, the moves are looped through and the first
    playable move is returned.

    Args:
        fen (str): FEN string to pass to the AI and validate moves from
        depth (int, optional): Amount of outputs/moves the AI should return. Defaults to 1.

    Raises:
        HTTPException: When FEN is invalid
        HTTPException: When there are no legal moves
        HTTPException: When the AI is not available

    Returns:
        str: The move in Algebraic Notation
    """
    # Catch malformed FEN
    try:
        board = chess.Board(fen)
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Invalid FEN code",
        )

    # Catch impossible move generation
    if board.legal_moves.count() == 0:
        raise HTTPException(
            status_code=403,
            detail="No moves possible"
        )

    # Catch AI errors
    try:
        ai_moves = train_model.generate_move(color=board.turn, fen=fen)
    except Exception:
        raise HTTPException(
            status_code=406,
            detail="AI not available"
        )

    # Get first legal move of outputs
    for ai_move in ai_moves:
        try:
            move = board.parse_uci(ai_move)
        except ValueError:
            continue
        if board.is_legal(move):
            return ai_move

    # Make legal_move generator a list, get the first move and then convert it to SAN
    # In other words: get legal move, even if bad
    return board.san(list(board.legal_moves)[0])
