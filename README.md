# Chess trainer
Train and play with your own chess bot using nevermind-neu and pleco.
Just follow simple instructions to train your own bot.

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

    `python <unpacked_pgn_file>`
    
5) Run training process for both sides(black and white) sequentially from project directory

    `cargo run --release train --dataset=py/chess_db_white.db --ocl --out=net_white --epochs=55`

    Then we need to train network evaluate positions from black side

    `cargo run --release train --dataset=py/chess_db_black.db --ocl --out=net_black --epochs=55`

    **--ocl** flag enables OpenCL computations on GPU
    **--epochs** spicifies number of epochs, could be modified.

    
6) Play with bot using some trained snapshot

    `cargo run --release play --state_white=<net_white...state> --state_black=<net_black...state> --ocl --unicode --depth=4`

    **--unicode** flag enables pretty unicode board state displaying

    **--depth** specifies the depth of move search for alpha-beta algorithm. I suggest to use values from 1 to 4. Big depth values(>4) will make the algorithm take a lot of time to search best move.
    
## GIF
![demo](https://github.com/regular-dev/chess_trainer/blob/master/doc/demo1.gif?raw=true)