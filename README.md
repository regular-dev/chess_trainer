# Chess trainer
Train and play with your own chess bot using nevermind-neu and pleco.
Just follow simple instruction to train your own bot.
## Usage
1) Clone repository
2) Run `cargo build --release`
3) Download any lichess pgn database from https://database.lichess.org/ (.pgn.zst) to chess_trainer/py folder, i suggest to choose not large file, for example "2014 - January" - 100 mb.
`cd py`
`wget https://database.lichess.org/standard/lichess_db_standard_rated_2014-01.pgn.zst`
`unzstd lichess_db_standard_rated_2014-01.pgn.zst` - unpack zstd archive
4) Convert pgn file to sqlite3 database with columns - [ fen , stockfish eval ] with python code. For engine evaluation you need to download stockfish engine binary. 
For ArchLinux it could be installed from AUR https://aur.archlinux.org/packages/stockfish. 
Otherwise you could download it from official site and provide path to stockfish-binary in **pgn_to_db.py**. 
    `python <unpacked_pgn_file> chess.db`
5) Run training process from project directory
    `cargo run --release train --dataset=py/chess.db --ocl`
    **--ocl** flag enables OpenCL computations on GPU
6) Play with bot using some trained snapshot
    `cargo run --release play --state=<some_state_file> --ocl --depth=4`
## GIF Demonstration