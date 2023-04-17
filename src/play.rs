use serde::Serialize;

use clap::ArgMatches;
use nevermind_neu::{
    models::*,
    orchestra::Orchestra,
};
use pleco::board::*;

use rand::Rng;

use std::{error::Error, io, process::Stdio};

use crate::train::*;
use crate::test::*;

pub fn play_chess(args: &ArgMatches) -> Result<(), Box<dyn Error>> {
    let file_path = args.get_one::<String>("ModelState").unwrap();
    let is_ocl = args.contains_id("Ocl");
    let is_fen = args.contains_id("Fen");
    let depth = args.get_one::<u16>("Depth").unwrap();

    if is_ocl {
        println!("Using ocl...");
        play_chess_ocl(file_path.clone(), is_fen, *depth)?;
    } else {
        play_chess_cpu(file_path.clone(), is_fen, *depth)?;
    }

    Ok(())
}

fn play_chess_ocl(model_state: String, display_fen: bool, d: u16) -> Result<(), Box<dyn Error>> {
    let mut mdl = SequentialOcl::new().expect("Failed to create SequentialOCL model");
    fill_ocl_model_with_layers(&mut mdl);
    mdl.load_state(&model_state)?;

    continue_play(mdl, display_fen, d)
}

fn play_chess_cpu(model_state: String, display_fen: bool, d: u16) -> Result<(), Box<dyn Error>> {
    let mut mdl = Sequential::new();
    fill_model_with_layers(&mut mdl);
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

fn continue_play<T: Model + Serialize + Clone>(mdl: T, display_fen: bool, d: u16) -> Result<(), Box<dyn Error>> {
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

    while !board.checkmate() {
        println!("==== Move {} ====", board.moves_played());

        if display_fen {
            println!("Fen : {}", board.fen());
        }

        board.pretty_print();

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

fn do_player_step(io: &io::Stdin, b: &mut Board) -> Result<(), Box<dyn Error>> 
{
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

fn do_bot_step<T: Model + Serialize + Clone>(b: &mut Board, orc: &mut Orchestra<T>, depth: u16) -> Result<(), Box<dyn Error>> {
    if b.moves_played() < 5 { // first 2 moves are random
        let rand_moves = b.generate_moves();
        let mut rng = rand::thread_rng();
        b.apply_move(rand_moves[rng.gen_range(0..rand_moves.len()) as usize]);
        return Ok(());
    } else {
        // let best_move = my_minimax(b, depth, orc, b.turn() == pleco::Player::White);
        let best_move = my_alpha_beta_search(b, -12000, 12000, depth, orc, b.turn() == pleco::Player::Black);
        b.apply_move(best_move.bit_move);
    }

    println!("Bot's move : {}", b.last_move().unwrap());

    Ok(())
}
