import train_model
import chess as c


def cli_interaction(color: c.COLORS):
    """
    Determine which interaction algorithm to use based on the color of the AI.
    :param color: color the AI is playing with
    """
    if color == c.WHITE:
        cli_interaction_white()
    else:
        cli_interaction_black()


def cli_interaction_black():
    """Interaction pattern when playing with the white pieces against the AI with the black pieces."""
    game_running = True
    board = c.Board()
    # printing the general formatting rules
    print("Please express your moves by stating the square of the piece you want to move and the square "
          "you wan to move to e.g. 'c2c4'!")
    print(board)
    while game_running:
        # getting the users move
        move = get_valid_move(board)
        board.push(move)
        print(board)
        if board.is_checkmate():
            game_running = False
        # Generating the AIs move
        pred = train_model.generate_move(c.BLACK, board.fen())
        board.push_san(pred)
        print(board)
        if board.is_checkmate():
            game_running = False
            break


def cli_interaction_white():
    """Interaction pattern when playing with the black pieces against the AI with the white pieces."""
    game_running = True
    board = c.Board()
    # printing the general formatting rules
    print("Please express your moves by stating the square of the piece you want to move and the square "
          "you wan to move to e.g. 'c2c4'!")
    print(board)
    while game_running:
        # Generating the AIs move
        pred = train_model.generate_move(c.WHITE, board.fen())
        board.push_san(pred)
        print(board)
        if board.is_checkmate():
            game_running = False
            break
        # getting the users move
        move = get_valid_move(board)
        board.push(move)
        print(board)
        if board.is_checkmate():
            game_running = False


def get_valid_move(board: c.Board):
    """
    Asks for a move to a given chess position until a valid one was given.
    :param board: the current position
    """
    move_invalid = True
    move = None
    while move_invalid:
        move = input("Insert your next move: ")

        try:
            # parse move
            move = c.Move.from_uci(move)
            # check if move is legal
            legal_moves_lst = [m for m in board.legal_moves]
            if move in legal_moves_lst: move_invalid = False
            else: print("Invalid Move!")

        except c.InvalidMoveError:
            print("Invalid Move!")

    return move
