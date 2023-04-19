use clap::ArgMatches;
use log::info;

use serde::Serialize;
use std::error::Error;

use nevermind_neu::models::*;

use pleco::Board;
use pleco::*;

use nevermind_neu::orchestra::*;

use crate::sqlite_dataset::*;
use crate::train::*;

const MATE_V: i16 = 31000 as i16;
const DRAW_V: i16 = 0 as i16;

pub fn test(args: &ArgMatches) -> Result<(), Box<dyn Error>> {
    let solver_state_str = args.get_one::<String>("ModelState").unwrap();
    let mut mdl = Sequential::new();
    fill_model_with_layers(&mut mdl);

    mdl.load_state(solver_state_str)?;

    continue_test(args, mdl)
}

pub fn test_ocl(args: &ArgMatches) -> Result<(), Box<dyn Error>> {
    let solver_state_str = args.get_one::<String>("ModelState").unwrap();
    let mut mdl = SequentialOcl::new().expect("Failed to create SequentialOCL model");
    fill_ocl_model_with_layers(&mut mdl);

    mdl.load_state(solver_state_str)?;

    continue_test(args, mdl)
}

pub fn continue_test<T: Model + Serialize + Clone>(
    args: &ArgMatches,
    mdl: T,
) -> Result<(), Box<dyn Error>> {
    let fen_str = args.get_one::<String>("Fen").unwrap();

    let mut board = Board::from_fen(fen_str.as_str()).unwrap();
    let mut vec_possible = Vec::new();

    vec_possible.push(encode_board(&mut board, 0.0).unwrap());

    let mut net = Orchestra::new_for_eval(mdl).test_batch_size(1);

    let out = net
        .eval_one(encode_board(&mut board, 0.0).unwrap().input)
        .unwrap();
    let out_b = out.borrow();

    info!("Current board eval : {}", out_b.first().unwrap());

    drop(out_b);

    let inv_val = board.turn() == pleco::Player::Black;
    let best_move = my_alpha_beta_search(&mut board, -14000, 14000, 2, &mut net, inv_val);
    // let best_move = my_minimax(&mut board, 2, &mut net);

    info!("Best move : {} - {}", best_move.bit_move, best_move.score);

    Ok(())
}

pub fn my_minimax<T: Model + Serialize + Clone>(
    board: &mut Board,
    depth: u16,
    net: &mut Orchestra<T>,
    black_or_white: bool, // black - false, white - true
) -> ScoringMove {
    if depth == 0 {
        let enc_b = encode_board(board, 0.0).unwrap();
        let out_net = net.eval_one(enc_b.input).unwrap();
        let out_net_b = out_net.borrow();
        let score = out_net_b.first().unwrap() * 15000.0;
        let score_move = ScoringMove::new_score(BitMove::new(0), score as i16);
        return score_move;
    }

    let mapped_vals = board
        .generate_scoring_moves()
        .into_iter()
        .map(|mut m: ScoringMove| {
            board.apply_move(m.bit_move);
            m.score = -my_minimax(board, depth - 1, net, black_or_white).score;
            board.undo_move();
            m
        });

    if black_or_white {
        return mapped_vals.min().unwrap_or_else(|| match board.in_check() {
            true => ScoringMove::blank(-MATE_V),
            false => ScoringMove::blank(DRAW_V),
        });
    } else {
        return mapped_vals.max().unwrap_or_else(|| match board.in_check() {
            true => ScoringMove::blank(-MATE_V),
            false => ScoringMove::blank(DRAW_V),
        });
    }
}

pub fn my_alpha_beta_search<T: Model + Serialize + Clone>(
    board: &mut Board,
    mut alpha: i16,
    beta: i16,
    depth: u16,
    net: &mut Orchestra<T>,
    inverse_value: bool, // false - for white, true for black
) -> ScoringMove {
    if depth == 0 {
        let enc_b = encode_board(board, 0.0).unwrap();
        let out_net = net.eval_one(enc_b.input).unwrap();
        let out_net_b = out_net.borrow();
        let mut score = out_net_b.first().unwrap() * 15000.0;

        if inverse_value {
            score = score * -1.0;
        }

        let score_move = ScoringMove::new_score(BitMove::new(0), score as i16);

        return score_move;
    }

    let mut moves = board.generate_scoring_moves();

    if moves.is_empty() {
        if board.in_check() {
            return ScoringMove::blank(-MATE_V);
        } else {
            return ScoringMove::blank(DRAW_V);
        }
    }

    let mut best_move = ScoringMove::blank(alpha);
    for mov in moves.iter_mut() {
        board.apply_move(mov.bit_move);
        mov.score =
            -my_alpha_beta_search(board, -beta, -alpha, depth - 1, net, inverse_value).score;
        board.undo_move();

        if mov.score > alpha {
            alpha = mov.score;
            if alpha >= beta {
                return *mov;
            }
            best_move = *mov;
        }
    }

    best_move
}
