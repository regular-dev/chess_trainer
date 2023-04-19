use clap::ArgMatches;
use pleco::*;

use rand::seq::SliceRandom;
use rand::thread_rng;
use nevermind_neu::dataloader::*;

use ndarray::array;
use ndarray::Array;

use log::{debug, error, info};

use crate::util;

const SCORE_LIMIT: f32 = 15.0;

pub fn dataset_from_db(args: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = args.get_one::<String>("DbPath").unwrap();
    let limit = args.get_one::<usize>("LimitDesk").unwrap();
    let db_offset = args.get_one::<usize>("Offset").unwrap();

    let mut loader = ProtobufDataLoader::empty();
    let connection = rusqlite::Connection::open(db_path).unwrap();

    let query = format!("SELECT * from evaluations LIMIT {} OFFSET {}", limit, db_offset);
    let mut stmt = connection.prepare(&query).unwrap();
    let mut rows = stmt.query([]).unwrap();

    let mut cnt = 0;

    while let Some(row) = rows.next().unwrap()
    {
        let fen:String = row.get_unwrap(1);
        let mut eval:f64 = row.get_unwrap(3);

        if eval > SCORE_LIMIT as f64 {
            eval = SCORE_LIMIT as f64;
        }

        if eval < -SCORE_LIMIT as f64 {
            eval = -SCORE_LIMIT as f64;
        }

        eval = (eval + 15.0) / 30.0;

        let board_opt = Board::from_fen(fen.as_str());

        if board_opt.is_err() {
            continue;
        }

        let mut board = board_opt.unwrap();

        let enc_board = encode_board(&mut board, eval as f32);

        if let Some(b) = enc_board {
            if cnt != 0 && cnt % 500 == 0 {
                info!("Pushed {} entries...", cnt);
                info!("Last eval : {}", eval);
            }

            loader.data.push(b);
            cnt += 1;
        }
    }

    let mut rng = thread_rng();

    loader.data.shuffle(&mut rng);

    loader.to_file("chess_data.db")?;

    Ok(())
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

pub fn encode_board(b: &mut Board, eval: f32) -> Option<LabeledEntry> {
    let mut db = LabeledEntry::default();

    let mut inp_vec = Vec::with_capacity(8 * 8 * 12);
    let mut legal_moves_white_v = Vec::with_capacity(8 * 8);
    let mut legal_moves_black_v = Vec::with_capacity(8 * 8);
    legal_moves_white_v.resize(8 * 8, 0.0);
    legal_moves_black_v.resize(8 * 8, 0.0);

    let mut flag_applied_null_move = false;

    for i in 0..8 as u8 {
        for j in 0..8 as u8 {
            let sq = SQ::from(7 - j + (8 * i));
            let p = b.piece_at_sq(sq);
            let piece_type = piece_to_type(&p);
            // let piece_val = piece_to_f32(&p);

            // inp_vec.push(piece_val);
            let mut piece_arr = Vec::with_capacity(12);
            piece_arr.resize(12, 0.0);

            if piece_type != 13 {
                piece_arr[piece_type] = 1.0;
            }
            inp_vec.append(&mut piece_arr);
        }
    }

    let legal_moves = b.generate_moves();

    // white possible moves
    debug!("legal {} {} moves...", b.turn(), legal_moves.len());
    for i in legal_moves.iter() {
        debug!("{} move : {} - {}", b.turn(), i, i.get_dest_u8());
        legal_moves_white_v[i.get_dest_u8() as usize] = 1.0;
    }

    if !b.checkmate() && b.checkers().is_empty() {
        unsafe {
            b.apply_null_move(); // switch player
            flag_applied_null_move = true;
        }
    
        // black legal moves
        let legal_moves = b.generate_moves();
    
        debug!("legal {} {} moves...", b.turn(), legal_moves.len());
        for i in legal_moves.iter() {
            debug!("{} move : {} - {}", b.turn(), i, i.get_dest_u8());
            legal_moves_black_v[i.get_dest_u8() as usize] = 1.0;
        }
    } else {
        for c in b.checkers().into_iter() {
            legal_moves_black_v[c.0 as usize] = 1.0;
        }
    }

    inp_vec.reverse();

    debug!(
        "legal moves white : {}",
        ndarray::Array::from_shape_vec(legal_moves_white_v.len(), legal_moves_white_v.clone())
            .unwrap()
    );
    debug!(
        "legal moves black : {}",
        ndarray::Array::from_shape_vec(legal_moves_black_v.len(), legal_moves_black_v.clone())
            .unwrap()
    );
    inp_vec.append(&mut legal_moves_white_v);
    inp_vec.append(&mut legal_moves_black_v);

    debug!(
        "score material black : {}",
        util::score_material(&b, Player::Black)
    );
    debug!(
        "score material white : {}",
        util::score_material(&b, Player::White)
    );

    // TODO : remove score_material values,
    // when dropout_ocl and sqlite_loader will be implemented
    inp_vec.push(util::score_material(&b, Player::Black)); 
    inp_vec.push(util::score_material(&b, Player::White));

    if flag_applied_null_move {
        unsafe {
            b.undo_null_move();
        }
    }

    if b.turn() == Player::White {
        inp_vec.push(1.0);
        inp_vec.push(0.0);
    } else {
        inp_vec.push(0.0);
        inp_vec.push(1.0);
    }

    let arr = Array::from_shape_vec(inp_vec.len(), inp_vec);

    if let Ok(a) = arr {
        db.input = a;
    } else {
        eprintln!("ERROR from shape vec !!!");
        return None;
    }

    db.expected = array![eval as f32];

    if eval.is_nan() {
        error!("NaN Value!");
        return None;
    }

    debug!("============== below ==================");
    debug!("fen : {}", b.fen());
    debug!("eval cur_board : {}", eval);
    debug!("delta raw : {}", eval);

    Some(db)
}