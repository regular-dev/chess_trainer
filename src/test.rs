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
    let state_white = args.get_one::<String>("ModelStateWhite").unwrap();
    let state_black = args.get_one::<String>("ModelStateBlack").unwrap();

    let mut mdl_white = Sequential::new();
    let mut mdl_black = Sequential::new();

    fill_model_with_layers(&mut mdl_white, false);
    fill_model_with_layers(&mut mdl_black, false);

    mdl_white.load_state(state_white)?;
    mdl_black.load_state(state_black)?;

    continue_test(args, mdl_white, mdl_black)
}

pub fn test_ocl(args: &ArgMatches) -> Result<(), Box<dyn Error>> {
    let state_white = args.get_one::<String>("ModelStateWhite").unwrap();
    let state_black = args.get_one::<String>("ModelStateBlack").unwrap();

    let mut mdl_white = SequentialOcl::new()?;
    let mut mdl_black = SequentialOcl::new()?;

    fill_ocl_model_with_layers(&mut mdl_white, false);
    fill_ocl_model_with_layers(&mut mdl_black, false);

    mdl_white.load_state(state_white)?;
    mdl_black.load_state(state_black)?;

    continue_test(args, mdl_white, mdl_black)
}

pub fn continue_test<T: Model + Serialize + Clone>(
    args: &ArgMatches,
    mdl_white: T,
    mdl_black: T,
) -> Result<(), Box<dyn Error>> {
    let fen_str = args.get_one::<String>("Fen").unwrap();

    let mut board = Board::from_fen(fen_str.as_str()).unwrap();
    let mut vec_possible = Vec::new();

    vec_possible.push(encode_board(&mut board, 0.0).unwrap());

    let mut net = Orchestra::new_for_eval(mdl_white).test_batch_size(1);
    let mut net_black = Orchestra::new_for_eval(mdl_black).test_batch_size(1);

    if board.turn() == pleco::Player::White {
        let out = net
            .eval_one(encode_board(&mut board, 0.0).unwrap().input)
            .unwrap();
        let out_b = out.borrow();

        info!("Current board eval : {}", out_b.first().unwrap());
    } else {
        let out = net_black
            .eval_one(encode_board(&mut board, 0.0).unwrap().input)
            .unwrap();
        let out_b = out.borrow();

        info!("Current board eval : {}", out_b.first().unwrap());
    }

    let depth = 4;
    let is_inv = if board.turn() == pleco::Player::White {
        depth % 2 == 1
    } else {
        depth % 2 == 0
    };

    let best_move = my_alpha_beta_search(
        &mut board,
        -15000,
        15000,
        depth,
        &mut net,
        &mut net_black,
        is_inv,
    );
    // let best_move = my_minimax(&mut board, 2, &mut net);

    info!("Best move : {} - {}", best_move.bit_move, best_move.score);

    Ok(())
}

pub fn my_minimax<T: Model + Serialize + Clone>(
    board: &mut Board,
    depth: u16,
    net: &mut Orchestra<T>,
    net_black: &mut Orchestra<T>,
    inv_val: bool,
) -> ScoringMove {
    if depth == 0 {
        let enc_b = encode_board(board, 0.0).unwrap();
        let mut score_move;

        if board.turn() == pleco::Player::White {
            let out_net = net.eval_one(enc_b.input).unwrap();
            let out_net_b = out_net.borrow();
            let score = (out_net_b.first().unwrap() - 0.5) * 15000.0;
            score_move = ScoringMove::new_score(BitMove::new(0), score as i16);
        } else {
            let out_net = net_black.eval_one(enc_b.input).unwrap();
            let out_net_b = out_net.borrow();
            let score = (out_net_b.first().unwrap() - 0.5) * 15000.0;
            score_move = ScoringMove::new_score(BitMove::new(0), score as i16);
        }

        if inv_val {
            score_move.score = -1 * score_move.score;
        }

        return score_move;
    }

    let mapped_vals = board
        .generate_scoring_moves()
        .into_iter()
        .map(|mut m: ScoringMove| {
            board.apply_move(m.bit_move);
            m.score = -my_minimax(board, depth - 1, net, net_black, inv_val).score;
            board.undo_move();
            m
        });

    return mapped_vals.max().unwrap_or_else(|| match board.in_check() {
        true => ScoringMove::blank(-MATE_V),
        false => ScoringMove::blank(DRAW_V),
    });
}

pub fn my_alpha_beta_search<T: Model + Serialize + Clone>(
    board: &mut Board,
    mut alpha: i16,
    beta: i16,
    depth: u16,
    net: &mut Orchestra<T>,
    net_black: &mut Orchestra<T>,
    inv_val: bool,
) -> ScoringMove {
    if depth == 0 {
        let enc_b = encode_board(board, 0.0).unwrap();
        let mut score_move;

        if board.turn() == pleco::Player::White {
            let out_net = net.eval_one(enc_b.input).unwrap();
            let out_net_b = out_net.borrow();
            let score = (out_net_b.first().unwrap() - 0.5) * 15000.0;
            score_move = ScoringMove::new_score(BitMove::new(0), score as i16);
        } else {
            let out_net = net_black.eval_one(enc_b.input).unwrap();
            let out_net_b = out_net.borrow();
            let score = (out_net_b.first().unwrap() - 0.5) * 15000.0;
            score_move = ScoringMove::new_score(BitMove::new(0), score as i16);
        }

        if inv_val {
            score_move.score = -1 * score_move.score;
        }

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
            -my_alpha_beta_search(board, -beta, -alpha, depth - 1, net, net_black, inv_val).score;
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
