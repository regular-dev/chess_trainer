import chess.pgn
import chess.engine
import sqlite3
import subprocess
import sys

if len(sys.argv) != 3:
    print("Invalid number of arguments!")
    exit(0)

__games_limit = 5000 # you can change this number if you want to increase dataset
_pgn_file = sys.argv[1].strip()
_out_db_file = sys.argv[2].strip()
_engine_path = 'stockfish'


pgn_file = open(_pgn_file)
conn = sqlite3.connect(_out_db_file)
c = conn.cursor()

# sqlite3 db schema
c.execute('''CREATE TABLE positions
             (fen text PRIMARY KEY, evaluation real)''')

game = chess.pgn.read_game(pgn_file)
_idx = 0

# init stockfish
engine = chess.engine.SimpleEngine.popen_uci(_engine_path)

while game and _idx < __games_limit:
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        fen = board.fen()
        evaluation = 0.0

        analysis = engine.analyse(board, chess.engine.Limit(depth=8))

        score = analysis["score"].white().score(mate_score=10000) / 100.0

        # clamp -25.0 to 25.0
        score = max(-25.0, min(25.0, score))

        evaluation = score

        try:
            c.execute("INSERT INTO positions VALUES (?, ?)", (fen, evaluation))
        except sqlite3.IntegrityError:
            pass

    if _idx % 100 == 0 and _idx != 0:
        print("Done {} games...".format(_idx))

    _idx += 1

    game = chess.pgn.read_game(pgn_file)

conn.commit()
conn.close()

print("Database finished...")
engine.quit()