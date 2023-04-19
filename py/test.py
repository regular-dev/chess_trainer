import chess.engine

# path to the Stockfish engine executable
engine_path = "stockfish"

# create the engine object
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# set the FEN position 
board_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
board = chess.Board(fen=board_fen)

# set the analysis parameters: limit the depth and time
depth = 8
time = 2.0

# perform the analysis
analysis = engine.analyse(board, chess.engine.Limit(depth=8))

# normalize the score to the range [-25.0, 25.0]
score_white = analysis["score"].white().score(mate_score=10000) / 100.0
score_black = analysis["score"].black().score(mate_score=10000) / 100.0

score_white = max(-25.0, min(25.0, score_white)) # clamp
score_black = max(-25.0, min(25.0, score_black)) # clamp

# print the score
print(score_white)
print(score_black)