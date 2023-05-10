use pleco::*;

const PRICE_PAWN: f32 = 1.0;
const PRICE_KNIGHT: f32 = 3.0;
const PRICE_BISHOP: f32 = 3.0;
const PRICE_ROOK: f32 = 5.0;
const PRICE_QUEEN: f32 = 10.0;
const MAX_PRICE: f32 = PRICE_PAWN * 8.0 + PRICE_KNIGHT * 2.0 + PRICE_BISHOP * 2.0 + PRICE_ROOK * 2.0 + PRICE_QUEEN;

pub fn score_material(b: &Board, p: Player) -> f32 {
    let pawn_cnt = b.count_piece(p, PieceType::P) as f32;
    let knight_cnt = b.count_piece(p, PieceType::K) as f32;
    let bishop_cnt = b.count_piece(p, PieceType::B) as f32;
    let rook_cnt = b.count_piece(p, PieceType::R) as f32;
    let queen_cnt = b.count_piece(p, PieceType::Q) as f32;

    let mut s = pawn_cnt * PRICE_PAWN
        + knight_cnt * PRICE_KNIGHT
        + bishop_cnt * PRICE_BISHOP
        + rook_cnt * PRICE_ROOK
        + queen_cnt
        + PRICE_QUEEN;

    if s > MAX_PRICE {
        s = MAX_PRICE;
    }

    s = s / MAX_PRICE;

    return s;
}
