/* --- WIP file --- */
use clap::ArgMatches;
use nevermind_neu::dataloader::*;
use nevermind_neu::util::*;
use pleco::bots::minimax::minimax;
use pleco::BitMove;
use pleco::MoveList;
use rand::seq::SliceRandom;

use ndarray::array;
use ndarray::{Array, Array2, ArrayView, ArrayViewMut, ArrayViewMut2, ShapeBuilder};

use std::fs::File;
use std::io;
use std::io::Write;
use std::ops::DerefMut;

use log::{debug, error, info, warn};

use rand::thread_rng;
use rand::Rng;

use pleco::bots::alphabeta;

use pleco::board::{Board, RandBoard};
use pleco::bots::alphabeta::*;
use pleco::core::Piece;
use pleco::core::Player;
use pleco::tools::eval::Eval;
use pleco::SQ;

const MAX_DESK_STEPS: i32 = 45;
const MITTELSPIEL: u16 = 23;
const DEPTH_SEARCH: usize = 4;
const RANDOM_MOVES: usize = 5;
const COEF_NORM: f32 = 10.0;
const SCORE_DIFF_PERC: f32 = 0.20; // 20%
const VAL_DIFF: f32 = 23.0;
const PIECE_TYPES: usize = 12;
const MAX_SCORE_DELTA: usize = 1200;

fn piece_to_f32(p: &Piece, turn: &Player) -> f32 {
    let out = match p {
        Piece::None => 0.0,

        // White figures
        Piece::WhitePawn => 1.0,
        Piece::WhiteKnight => 3.0,
        Piece::WhiteBishop => 4.0,
        Piece::WhiteRook => 5.0,
        Piece::WhiteQueen => 10.0,
        Piece::WhiteKing => 20.0,

        // Black figuresa
        Piece::BlackPawn => -1.0,
        Piece::BlackKnight => -3.0,
        Piece::BlackBishop => -4.0,
        Piece::BlackRook => -5.0,
        Piece::BlackQueen => -10.0,
        Piece::BlackKing => -20.0,
        //_ => 0.0,
    };

    if *turn == Player::Black {
        if out != 0.0 {
            return out * -1.0;
        } else {
            return 0.0;
        }
    }

    return out;
}

fn piece_to_type(p: &Piece) -> usize {
    match p {
        Piece::None => 13,

        Piece::WhitePawn => 0,
        Piece::WhiteKnight => 1,
        Piece::WhiteBishop => 2,
        Piece::WhiteRook => 3,
        Piece::WhiteQueen => 4,
        Piece::WhiteKing => 5,

        Piece::BlackPawn => 6,
        Piece::BlackKnight => 7,
        Piece::BlackBishop => 8,
        Piece::BlackRook => 9,
        Piece::BlackQueen => 10,
        Piece::BlackKing => 11,
    }
}

pub unsafe fn flip_board(board: &mut ArrayViewMut2<f32>) -> Array2<f32> {
    let mut out_arr = Array2::<f32>::zeros((8, 8));

    for i in 0..8 {
        for j in 0..8 {
            *out_arr.uget_mut((i, j)) = *board.uget((7 - i, 7 - j)); //*board.uget_mut((i, j));
                                                                     //*out_arr.uget_mut((i, 7 - j)) = *board.uget((7 - i, j));
        }
    }

    out_arr
}

/// If we playing white remain board the same
/// If we playing black rotate board 180
pub fn encode_to_databatch(board: &Board, val: i32) -> Option<LabeledEntry> {
    let mut db = LabeledEntry::default();

    let mut inp_vec = Vec::new();
    let turn = board.turn();

    for i in 0..8 as u8 {
        for j in 0..8 as u8 {
            let sq = SQ::from(7 - j + (8 * i));
            let p = board.piece_at_sq(sq);
            inp_vec.push(piece_to_f32(&p, &turn));
        }
    }

    inp_vec.reverse();

    let arr = Array::from_shape_vec(inp_vec.len(), inp_vec);

    if let Ok(mut a) = arr {
        if let Ok(mut arr_view) = ArrayViewMut::from_shape((8, 8), a.as_slice_mut().unwrap()) {
            if turn == Player::Black {
                unsafe {
                    if let Ok(flipped_board) = flip_board(&mut arr_view).into_shape(8 * 8) {
                        db.input = flipped_board;
                    }
                }
            } else {
                db.input = a;
            }
        }
    } else {
        eprintln!("ERROR from shape vec !!!");
        return None;
    }

    db.expected = array![val as f32];

    minmax_normalize_params(&mut db.input, -20.0, 20.0);
    minmax_normalize_params(&mut db.expected, -6700.0, 6700.0);

    // make output between -1.0 and 1.0

    db.input = db.input.map(|v| v * 2.0 - 1.0);

    let av = ArrayView::from_shape((8, 8), db.input.as_slice().unwrap()).unwrap();
    //println!("db.input : {}", &av);

    let exp_val = db.expected[0].clone();
    *db.expected.get_mut(0).unwrap() = (exp_val * 2.0) - 1.0;

    //println!("normalized output value : {}", db.expected[0]);

    Some(db)
}

fn generate_n_random(n: usize, max_val: usize) -> Vec<usize> {
    if max_val < n {
        panic!("Invalid use of generate_n_random");
    }

    let mut rng = rand::thread_rng();

    unsafe {
        static mut v: Vec<usize> = Vec::new();

        v.clear();

        for i in 0..max_val {
            v.push(i)
        }

        v.shuffle(&mut rng);

        let mut v_out = Vec::with_capacity(n);

        for i in 0..n {
            v_out.push(v[i]);
        }

        v_out
    }
}

pub fn encode_board_with_move(b: &Board, m: &BitMove, val: f32) -> Option<LabeledEntry> {
    let mut db = LabeledEntry::default();

    let mut inp_vec = Vec::with_capacity(8 * 8 * 12 + 8 * 8);
    let mut mov_vec = Vec::new();
    mov_vec.resize(8 * 8, 0.0);

    //let turn = b.turn();

    for i in 0..8 as u8 {
        for j in 0..8 as u8 {
            let sq = SQ::from(7 - j + (8 * i));
            let p = b.piece_at_sq(sq);
            let piece_type = piece_to_type(&p);

            let mut piece_arr = Vec::new();
            piece_arr.resize(12, 0.0);

            if piece_type != 13 {
                piece_arr[piece_type] = 1.0;
            }
            inp_vec.append(&mut piece_arr);
            //inp_vec.push(piece_to_f32(&p, &turn));
        }
    }

    inp_vec.reverse();

    // src
    {
        let src_x = m.get_src_u8() as usize / 8;
        let src_y = m.get_src_u8() as usize % 8;

        mov_vec[(7 - src_x) * 8 + src_y] = 1.0;
    }

    // dst
    {
        let dst_x = m.get_dest_u8() as usize / 8;
        let dst_y = m.get_dest_u8() as usize % 8;

        mov_vec[(7 - dst_x) * 8 + dst_y] = 1.0;
    }

    inp_vec.append(&mut mov_vec);

    let arr = Array::from_shape_vec(inp_vec.len(), inp_vec);

    if let Ok(a) = arr {
        db.input = a;
    } else {
        eprintln!("ERROR from shape vec !!!");
        return None;
    }

    db.expected = array![val as f32];

    if val.is_nan() {
        error!("NAN VALUE !!!!!");
    }

    //minmax_normalize_params(&mut db.input, -20.0, 20.0);
    //minmax_normalize_params(&mut db.expected, -1000.0, 1000.0);

    // make input between -1.0 and 1.0
    // db.input = db.input.map(|v| v * 2.0 - 1.0);

    debug!("============== below ==================");
    debug!("fen : {}", b.fen());
    debug!("eval cur_board : {}", Eval::eval_low(&b));
    debug!("delta raw : {}", val);

    debug!(
        "move src : {} | {} , move dst : {} | {}",
        m.get_src(),
        m.get_src_u8(),
        m.get_dest(),
        m.get_dest_u8()
    );

    //let av = ArrayView::from_shape((16, 8), db.input.as_slice().unwrap()).unwrap();
    //debug!("db.input : {}", &av);

    Some(db)
}

fn rate_move_list(b: &mut Board, movelist: &MoveList, depth: u16) -> Vec<i32> {
    let cur_board_eval = Eval::eval_low(b);
    let mut v = Vec::with_capacity(movelist.len());

    for i in movelist.iter() {
        let eval = alpha_beta_eval_bitmove(b, i.clone(), -14000, 14000, depth);
        v.push(eval as i32 - cur_board_eval);
    }

    v
}

fn minmax_normalize_vec(v: &Vec<i32>) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }

    let mut v_out = Vec::with_capacity(v.len());

    let vmax = *v.iter().max().unwrap() as f32;
    let vmin = *v.iter().min().unwrap() as f32;

    info!("vmin : {} , vmax : {}", vmin, vmax);

    let delta = vmax - vmin;

    for i in v.iter() {
        let mut out = (*i as f32 - vmin) / delta as f32;

        if out.is_nan() || out.is_infinite() {
            warn!("out is NAN setting 0.999 for values : {} - {}", vmax, vmin);
            // out = 0.999
            return Vec::new();
        }

        if out > 1.0 {
            out = 0.9999;
        }

        if out < 0.0 {
            out = 0.0;
        }

        if out == std::f32::EPSILON {
            warn!(
                "out is EPSILON setting to 0.0 for values : {} - {}",
                vmax, vmin
            );
            out = 0.0;
        }

        v_out.push(out * COEF_NORM);
    }

    v_out
}

fn zscore_normalize_vec(v: &Vec<i32>) -> Vec<f32> {
    let sum: i32 = v.iter().sum();
    let mean = sum as f32 / v.len() as f32;

    let mut differences = Vec::with_capacity(v.len());
    for i in v.iter() {
        differences.push((*i as f32 - mean).powf(2.0));
    }
    let sum_diff: f32 = differences.iter().sum();
    let standart_deviation = (sum_diff as f32 / (v.len() as f32 - 1.0)).sqrt();

    let mut zscores = Vec::with_capacity(v.len());

    let mut file = File::create("zscore.txt").unwrap();
    file.write_fmt(format_args!(
        "mean : {} | standart_deviation : {}",
        mean, standart_deviation
    ));
    // Write a &str in the file (ignoring the result).
    // writeln!(&mut file, "Hello World!").unwrap();

    for i in v.iter() {
        zscores.push((*i as f32 - mean) / standart_deviation);
    }

    zscores
}

fn generate_score_bitmove_vec<T: Copy>(moves: &MoveList, scores: &Vec<T>) -> Vec<(BitMove, T)> {
    let mut v_out = Vec::with_capacity(moves.len());

    for i in 0..moves.len() {
        v_out.push((moves[i], scores[i]));
    }

    v_out
}

pub fn minmax_normalize_vec_batch(vec: &mut Vec<LabeledEntry>) {
    let mut comp_vec = Vec::with_capacity(vec.len());

    // let mut file_out_orig = File::create("outvals_orig.txt").unwrap();

    for i in vec.iter() {
        // file_out_orig.write_fmt(format_args!("{}\n", *i.expected.get(0).unwrap() as i32));
        comp_vec.push(*i.expected.get(0).unwrap() as i32);
    }

    let normalized = minmax_normalize_vec(&comp_vec);

    let mut file_out = File::create("outvals.txt").unwrap();

    for (i, norm) in vec.iter_mut().zip(normalized.iter()) {
        //file_out.write_fmt(format_args!("{}\n", norm));
        i.expected = array![*norm];
    }
}

fn remove_threshold(vec: &mut Vec<LabeledEntry>) {
    vec.retain(|b| (*b.expected.get(0).unwrap() as f32).abs() < MAX_SCORE_DELTA as f32);
}

pub fn create_dataset(filepath: &str, desk_num: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut loader = ProtobufDataLoader::empty();

    let mut rng = rand::thread_rng();

    info!("Generating {} random desks", desk_num);

    let mut boards: Vec<Board> = RandBoard::new()
        .pseudo_random(rng.gen_range(0..10000))
        .no_check()
        .from_start_pos()
        .many(desk_num);

    let mut counter: usize = 0;

    for (num_b, b) in boards.iter_mut().enumerate() {
        let mut iter_desk = 0;

        info!("Handling another {} desk", num_b);

        let mut prev_score = 0.0;

        while !b.checkmate() && iter_desk < MAX_DESK_STEPS {
            iter_desk += 1;

            let eval_cur_board = Eval::eval_low(&b) as i16;

            if b.turn() == Player::White {
                let mut legal_moves = b.generate_moves();
                legal_moves.shuffle(&mut rng);

                if legal_moves.is_empty() {
                    break;
                }

                let score_lm = rate_move_list(b, &legal_moves, DEPTH_SEARCH as u16 - 1);
                // let normalized_lm = minmax_normalize_vec(&score_lm);
                let normalized_lm = score_lm;

                if !normalized_lm.is_empty() {
                    let mut tuple_vec = generate_score_bitmove_vec(&legal_moves, &normalized_lm);

                    let rand_moves_num = std::cmp::min(RANDOM_MOVES, legal_moves.len());

                    tuple_vec.shuffle(&mut rng);

                    debug!("BEGIN {} DISPLAY --------", rand_moves_num);

                    for i in 0..rand_moves_num {
                        if rand_moves_num == 1 {
                            break;
                        }

                        if tuple_vec[i].0.is_castle() {
                            info!("Castling move - continue");
                            continue;
                        }

                        if (tuple_vec[i].1 as f32 - prev_score).abs()
                            < prev_score.abs() * SCORE_DIFF_PERC
                            || (tuple_vec[i].1 as f32 - prev_score).abs() < VAL_DIFF
                        {
                            let delta = (tuple_vec[i].1 as f32 - prev_score).abs();
                            debug!("Ignoring move with delta : {}", delta);
                            continue;
                        }

                        let db = encode_board_with_move(b, &tuple_vec[i].0, tuple_vec[i].1 as f32);

                        debug!("MOVE : {} | SCORE : {}", tuple_vec[i].0, tuple_vec[i].1);

                        if let Some(val_db) = db {
                            loader.data.push(val_db);
                            prev_score = tuple_vec[i].1 as f32;

                            if counter % 100 == 0 {
                                info!(
                                    "Successfully pushed {} databatches. Last score : {}",
                                    counter, tuple_vec[i].1
                                );
                            }
                            counter += 1;
                        } else {
                            error!("Something wrong with encoding databatch !!!");
                        }
                    }
                }
            }

            let best_move = alphabeta::alpha_beta_search(b, -14000, 14000, DEPTH_SEARCH as u16);
            // let best_move = minimax(b, DEPTH_SEARCH as u16);

            if b.turn() == Player::White {
                // encoding best move also
                let db = encode_board_with_move(
                    &b,
                    &best_move.bit_move,
                    best_move.score as f32 - eval_cur_board as f32,
                )
                .unwrap();
                loader.data.push(db);

                debug!(
                    "BEST MOVE AND SCORE : {}-{} : {}",
                    best_move.bit_move.get_src(),
                    best_move.bit_move.get_dest(),
                    best_move.score - eval_cur_board,
                );
                debug!("--- END MOVES ---");
            }

            if b.legal_move(best_move.bit_move) {
                b.apply_move(best_move.bit_move);
            } else {
                warn!("generated ILLEGAL MOVE !!!");
                break;
            }

            if iter_desk == MAX_DESK_STEPS {
                debug!("reached max desk steps...");
                break;
            }
        }
    }

    let mut rng = thread_rng();

    loader.data.shuffle(&mut rng);

    remove_threshold(&mut loader.data);
    minmax_normalize_vec_batch(&mut loader.data);

    loader.to_file(filepath)?;

    Ok(())
}

pub fn dataset_info(args: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let ds_path = args.get_one::<String>("Dataset").unwrap();

    let loader = ProtobufDataLoader::from_file(ds_path)?;
    println!("Dataset length : {}", loader.data.len());

    for i in loader.data {
        println!("Eval encoded : {}", i.expected.first().unwrap());
    }

    Ok(())
}