use pleco::Piece;
use pleco::ScoringMove;
use serde::Serialize;

use log::info;

use clap::ArgMatches;
use nevermind_neu::{models::*, orchestra::Orchestra};
use pleco::board::*;

use rand::Rng;

use pleco::{core::masks::SQ_DISPLAY_ORDER, SQ};

use std::{error::Error, io};

use crate::sqlite_dataset::encode_board;
use crate::test::*;
use crate::train::*;

pub fn play_chess(args: &ArgMatches) -> Result<(), Box<dyn Error>> {
    let file_path = args.get_one::<String>("ModelState").unwrap();
    let is_ocl = args.contains_id("Ocl");
    let is_fen = args.contains_id("Fen");
    let mut depth = args.get_one::<u16>("Depth").unwrap().clone();

    if depth == 0 {
        depth = 2;
    }

    if depth % 2 == 1 {
        depth += 1;
    }

    if is_ocl {
        info!("Using ocl...");
        play_chess_ocl(file_path.clone(), is_fen, depth)?;
    } else {
        play_chess_cpu(file_path.clone(), is_fen, depth)?;
    }

    Ok(())
}

fn play_chess_ocl(model_state: String, display_fen: bool, d: u16) -> Result<(), Box<dyn Error>> {
    let mut mdl = SequentialOcl::new().expect("Failed to create SequentialOCL model");
    fill_ocl_model_with_layers(&mut mdl, false);
    mdl.load_state(&model_state)?;

    continue_play(mdl, display_fen, d)
}

fn play_chess_cpu(model_state: String, display_fen: bool, d: u16) -> Result<(), Box<dyn Error>> {
    let mut mdl = Sequential::new();
    fill_model_with_layers(&mut mdl, false);
    mdl.load_state(&model_state)?;

    continue_play(mdl, display_fen, d)
}

fn read_string_from_stdin(stdin: &io::Stdin) -> Result<String, Box<dyn Error>> {
    let mut buf_str = String::new();
    stdin.read_line(&mut buf_str)?;
    buf_str.pop(); // remove \n
    Ok(buf_str)
}

#[derive(PartialEq)]
enum Turn {
    Player,
    PC,
}

impl Turn {
    pub fn switch(&mut self) {
        if *self == Turn::PC {
            *self = Turn::Player;
        } else {
            *self = Turn::PC;
        }
    }
}

fn continue_play<T: Model + Serialize + Clone>(
    mdl: T,
    display_fen: bool,
    d: u16,
) -> Result<(), Box<dyn Error>> {
    // initialize orchestra
    let mut orc = Orchestra::new_for_eval(mdl).test_batch_size(1);

    let stdin = io::stdin();

    println!("[B]lack or [W]hite ?");

    let side = read_string_from_stdin(&stdin)
        .expect("Failed to read side")
        .to_lowercase();

    let mut turn = Turn::Player;

    if side == "b" {
        turn = Turn::PC;
    }

    let mut board = Board::start_pos();

    println!("UCI Move examples : 'e2e4', 'e7e8q' - pawn to queen promotes");

    while !board.checkmate() {
        println!("==== Move {} ====", board.moves_played());

        if display_fen {
            println!("Fen : {}", board.fen());
        }

        print_board(&mut board);

        if turn == Turn::Player {
            do_player_step(&stdin, &mut board)?;
        } else {
            println!("Bot is thinking...");
            do_bot_step(&mut board, &mut orc, d)?;
        }

        turn.switch();
    }

    Ok(())
}

fn do_player_step(io: &io::Stdin, b: &mut Board) -> Result<(), Box<dyn Error>> {
    loop {
        let player_move = read_string_from_stdin(io)?;
        let result = b.apply_uci_move(&player_move);

        if result {
            break;
        }
        println!("Invalid UCI move notation, please try again");
    }

    Ok(())
}

fn do_bot_step<T: Model + Serialize + Clone>(
    b: &mut Board,
    orc: &mut Orchestra<T>,
    depth: u16,
) -> Result<(), Box<dyn Error>> {
    if b.moves_played() < 3 {
        // first 2 moves are random
        let rand_moves = b.generate_moves();
        let mut rng = rand::thread_rng();
        b.apply_move(rand_moves[rng.gen_range(0..rand_moves.len()) as usize]);
    } else if b.moves_played() < 5 {
        let best_move =
            my_alpha_beta_search(b, -14000, 14000, 2, orc, b.turn() == pleco::Player::Black);
        b.apply_move(best_move.bit_move);
    } else {
        // let best_move = my_minimax(b, depth, orc, b.turn() == pleco::Player::White);
        let best_move = my_alpha_beta_search(b, -15000, 15000, depth, orc, b.turn() == pleco::Player::Black);
        // let best_move = shorten_alpha_beta( TODO : impl
        //     b,a
        //     -14000 as f32,
        //     14000 as f32,
        //     depth,
        //     orc,
        //     b.turn() == pleco::Player::Black,
        //     6
        // );
        b.apply_move(best_move.bit_move);
    }

    println!("Bot's move : {}", b.last_move().unwrap());

    Ok(())
}

fn print_board(b: &mut Board) {
    let mut out_str = String::with_capacity(64 * 4);

    let top_heading = vec!['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
    let left_heading = vec!['1', '2', '3', '4', '5', '6', '7', '8'];

    out_str.push(' ');
    out_str.push(' ');

    for i in top_heading.iter() {
        out_str.push(*i);
        out_str.push(' ');
    }

    out_str.pop();
    out_str.push('\n');

    out_str.push(left_heading[7]);
    out_str.push(' ');

    let mut left_heading_idx = 1;

    for sq in SQ_DISPLAY_ORDER.iter() {
        let op = b.piece_at_sq(SQ(*sq));

        let char = if op != Piece::None {
            //op.character_lossy()
            piece_to_pretty_char(&op)
        } else {
            '-'
        };

        out_str.push(char);
        out_str.push(' ');

        if sq % 8 == 7 {
            out_str.push('\n');

            if left_heading_idx < 8 {
                out_str.push(left_heading[7 - left_heading_idx]);
                out_str.push(' ');
                left_heading_idx += 1;
            }
        }
    }

    println!("{}", out_str);
}

fn piece_to_pretty_char(p: &Piece) -> char {
    match p {
        Piece::None => panic!(),
        Piece::WhitePawn => '♙',
        Piece::WhiteKnight => '♘',
        Piece::WhiteBishop => '♗',
        Piece::WhiteRook => '♖',
        Piece::WhiteQueen => '♕',
        Piece::WhiteKing => '♔',
        Piece::BlackPawn => '♟',
        Piece::BlackKnight => '♞',
        Piece::BlackBishop => '♝',
        Piece::BlackRook => '♜',
        Piece::BlackQueen => '♛',
        Piece::BlackKing => '♚',
    }
}

pub fn generate_top_n_moves<T: Model + Serialize + Clone>(
    b: &mut Board,
    orc: &mut Orchestra<T>,
    n: usize,
) -> Vec<(pleco::BitMove, f32)> {
    let legal_moves = b.generate_moves();    

    let mut scored_moves:Vec<(pleco::BitMove, f32)> = legal_moves.iter().map(|m|
    {
        b.apply_move(m.clone());

        let enc_b = encode_board(b, 0.0).unwrap();
        let out_net = orc.eval_one(enc_b.input).unwrap();
        let out_net_b = out_net.borrow();
        let mut eval = *out_net_b.first().unwrap();

        b.undo_move();

        (m.clone(), *out_net_b.first().unwrap())

    }).collect();

    scored_moves.sort_by(|a, b|
    {
        // a.1.total_cmp(&b.1)
        a.1.partial_cmp(&b.1).unwrap()
    });

    scored_moves.truncate(n);

    scored_moves
}
