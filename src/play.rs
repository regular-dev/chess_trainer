use pleco::Piece;
use serde::Serialize;

use log::info;

use clap::ArgMatches;
use nevermind_neu::{models::*, orchestra::Orchestra};
use pleco::board::*;

use rand::Rng;

use pleco::{core::masks::SQ_DISPLAY_ORDER, SQ};

use std::{error::Error, io};

use crate::test::*;
use crate::train::*;

pub fn play_chess(args: &ArgMatches) -> Result<(), Box<dyn Error>> {
    let file_path_white = args.get_one::<String>("ModelStateWhite").unwrap();
    let file_path_black = args.get_one::<String>("ModelStateBlack").unwrap();
    let is_ocl = args.contains_id("Ocl");
    let is_fen = args.contains_id("Fen");
    let unicode = args.contains_id("UnicodeDisplay");
    let mut depth = args.get_one::<u16>("Depth").unwrap().clone();

    if depth == 0 {
        depth = 2;
    }

    if is_ocl {
        info!("Using ocl...");
        play_chess_ocl(
            file_path_white.clone(),
            file_path_black.clone(),
            is_fen,
            depth,
            unicode,
        )?;
    } else {
        play_chess_cpu(
            file_path_white.clone(),
            file_path_black.clone(),
            is_fen,
            depth,
            unicode,
        )?;
    }

    Ok(())
}

fn play_chess_ocl(
    model_state_white: String,
    model_state_black: String,
    display_fen: bool,
    d: u16,
    unicode: bool,
) -> Result<(), Box<dyn Error>> {
    let mut mdl_white = SequentialOcl::new()?;
    fill_ocl_model_with_layers(&mut mdl_white, false);
    mdl_white.load_state(&model_state_white)?;

    let mut mdl_black = SequentialOcl::new()?;
    fill_ocl_model_with_layers(&mut mdl_black, false);
    mdl_black.load_state(&model_state_black)?;

    continue_play(mdl_white, mdl_black, display_fen, d, unicode)
}

fn play_chess_cpu(
    model_state_white: String,
    model_state_black: String,
    display_fen: bool,
    d: u16,
    unicode: bool,
) -> Result<(), Box<dyn Error>> {
    let mut mdl_white = Sequential::new();
    fill_model_with_layers(&mut mdl_white, false);
    mdl_white.load_state(&model_state_white)?;

    let mut mdl_black = Sequential::new();
    fill_model_with_layers(&mut mdl_black, false);
    mdl_black.load_state(&model_state_black)?;

    continue_play(mdl_white, mdl_black, display_fen, d, unicode)
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
    mdl_white: T,
    mdl_black: T,
    display_fen: bool,
    d: u16,
    unicode: bool,
) -> Result<(), Box<dyn Error>> {
    // initialize orchestra
    let mut orc_white = Orchestra::new_for_eval(mdl_white).test_batch_size(1);
    let mut orc_black = Orchestra::new_for_eval(mdl_black).test_batch_size(1);

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

        print_board(&mut board, unicode);

        if turn == Turn::Player {
            do_player_step(&stdin, &mut board)?;
        } else {
            println!("Bot is thinking...");
            do_bot_step(&mut board, &mut orc_white, &mut orc_black, d)?;
        }

        turn.switch();
    }

    println!("Checkmate!");

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
    orc_black: &mut Orchestra<T>,
    depth: u16,
) -> Result<(), Box<dyn Error>> {
    if b.moves_played() < 2 {
        // first move is random
        let rand_moves = b.generate_moves();
        let mut rng = rand::thread_rng();
        b.apply_move(rand_moves[rng.gen_range(0..rand_moves.len()) as usize]);
    } else if b.moves_played() < 4 {
        let is_inv = if b.turn() == pleco::Player::White {
            depth % 2 == 1
        } else {
            depth % 2 == 0
        };
        let best_move = my_alpha_beta_search(b, -15000, 15000, 2, orc, orc_black, is_inv);
        b.apply_move(best_move.bit_move);
    } else {
        let is_inv = if b.turn() == pleco::Player::White {
            depth % 2 == 1
        } else {
            depth % 2 == 0
        };
        let best_move = my_alpha_beta_search(b, -16000, 16000, depth, orc, orc_black, is_inv);
        b.apply_move(best_move.bit_move);
    }

    println!("Bot's move : {}", b.last_move().unwrap());

    Ok(())
}

fn print_board(b: &mut Board, unicode: bool) {
    let mut out_str = String::with_capacity(64 * 4);

    let top_heading = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
    let left_heading = ['1', '2', '3', '4', '5', '6', '7', '8'];

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
            if unicode {
                piece_to_pretty_char(&op)
            } else {
                piece_to_simple_char(&op)
            }
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

fn piece_to_simple_char(p: &Piece) -> char {
    match p {
        Piece::None => panic!(),
        Piece::WhitePawn => 'P',
        Piece::WhiteKnight => 'N',
        Piece::WhiteBishop => 'B',
        Piece::WhiteRook => 'R',
        Piece::WhiteQueen => 'Q',
        Piece::WhiteKing => 'K',
        Piece::BlackPawn => 'p',
        Piece::BlackKnight => 'n',
        Piece::BlackBishop => 'b',
        Piece::BlackRook => 'r',
        Piece::BlackQueen => 'q',
        Piece::BlackKing => 'k',
    }
}
