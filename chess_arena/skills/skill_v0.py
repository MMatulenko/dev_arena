import chess

# Piece material values (centipawns / 100 = pawns)
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Central squares rewarded for occupancy / attacks
CENTER_SQUARES = {chess.E4, chess.D4, chess.E5, chess.D5}
EXTENDED_CENTER = {chess.C3, chess.D3, chess.E3, chess.F3,
                   chess.C4, chess.F4, chess.C5, chess.F5,
                   chess.C6, chess.D6, chess.E6, chess.F6}


def _material(board: chess.Board) -> float:
    score = 0.0
    for piece_type, val in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * val
        score -= len(board.pieces(piece_type, chess.BLACK)) * val
    return score


def _center_control(board: chess.Board) -> float:
    """Reward pawns and pieces occupying or attacking the central squares."""
    score = 0.0
    for sq in CENTER_SQUARES:
        piece = board.piece_at(sq)
        if piece:
            bonus = 0.3 if piece.piece_type == chess.PAWN else 0.15
            score += bonus if piece.color == chess.WHITE else -bonus
        # Count attackers
        w_attackers = len(board.attackers(chess.WHITE, sq))
        b_attackers = len(board.attackers(chess.BLACK, sq))
        score += 0.05 * (w_attackers - b_attackers)

    for sq in EXTENDED_CENTER:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN:
            score += 0.1 if piece.color == chess.WHITE else -0.1
    return score


def _mobility(board: chess.Board) -> float:
    """Reward having more legal moves (= more options)."""
    if board.turn == chess.WHITE:
        w_moves = len(list(board.legal_moves))
        board.push(chess.Move.null())
        b_moves = len(list(board.legal_moves))
        board.pop()
    else:
        b_moves = len(list(board.legal_moves))
        board.push(chess.Move.null())
        w_moves = len(list(board.legal_moves))
        board.pop()
    return 0.05 * (w_moves - b_moves)


def _passed_pawns(board: chess.Board) -> float:
    """Reward passed pawns — pawns with no opposing pawn blocking their path."""
    score = 0.0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        file_ = chess.square_file(sq)
        rank = chess.square_rank(sq)
        blocked = False
        for opp_sq in board.pieces(chess.PAWN, chess.BLACK):
            opp_file = chess.square_file(opp_sq)
            opp_rank = chess.square_rank(opp_sq)
            if abs(opp_file - file_) <= 1 and opp_rank > rank:
                blocked = True
                break
        if not blocked:
            # Value passed pawn more the further advanced it is
            score += 0.1 * (rank - 1)

    for sq in board.pieces(chess.PAWN, chess.BLACK):
        file_ = chess.square_file(sq)
        rank = chess.square_rank(sq)
        blocked = False
        for opp_sq in board.pieces(chess.PAWN, chess.WHITE):
            opp_file = chess.square_file(opp_sq)
            opp_rank = chess.square_rank(opp_sq)
            if abs(opp_file - file_) <= 1 and opp_rank < rank:
                blocked = True
                break
        if not blocked:
            score -= 0.1 * (6 - rank)
    return score


def evaluate_board(board: chess.Board) -> float:
    """
    Baseline evaluation function combining:
      - Material count
      - Center control
      - Mobility
      - Passed pawn detection

    Positive = good for White, Negative = good for Black.
    """
    if board.is_checkmate():
        # The side that just moved delivered checkmate
        return -9999.0 if board.turn == chess.WHITE else 9999.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    score = _material(board)
    score += _center_control(board)

    try:
        score += _mobility(board)
    except Exception:
        pass  # null-move not always legal; skip mobility safely

    score += _passed_pawns(board)
    return score
