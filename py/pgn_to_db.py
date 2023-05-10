import chess.pgn
import chess.engine
import sqlite3
import subprocess
import sys

if len(sys.argv) != 2:
    print("Invalid number of arguments!")
    print("Usage python pgn_to_db.py <pgn_input_file>")
    exit(0)

__games_limit = 10000 # you can change this number if you want to increase dataset
_pgn_file = sys.argv[1].strip()
_out_db_file_white = 'chess_db_white.db'
_out_db_file_black = 'chess_db_black.db'
_engine_path = 'stockfish'

pgn_file = open(_pgn_file)
conn_white = sqlite3.connect(_out_db_file_white)
conn_black = sqlite3.connect(_out_db_file_black)
c_white = conn_white.cursor()
c_black = conn_black.cursor()

# sqlite3 db schema
c_white.execute('''CREATE TABLE positions
             (fen text PRIMARY KEY, evaluation real)''')
c_black.execute('''CREATE TABLE positions
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

        analysis = engine.analyse(board, chess.engine.Limit(depth=10))

        score = analysis["score"].white().score(mate_score=10000) / 100.0

        # clamp -25.0 to 25.0
        score = max(-25.0, min(25.0, score))

        evaluation = score

        try:
            if board.turn == chess.WHITE:
                c_white.execute("INSERT INTO positions VALUES (?, ?)", (fen, evaluation))
            else:
                c_black.execute("INSERT INTO positions VALUES (?, ?)", (fen, evaluation))
        except sqlite3.IntegrityError:
            pass
        

    if _idx % 100 == 0 and _idx != 0:
        print("Done {} games...".format(_idx))

    _idx += 1

    game = chess.pgn.read_game(pgn_file)

conn_white.commit()
conn_white.close()
conn_black.commit()
conn_black.close()

print("Database finished...")
engine.quit()