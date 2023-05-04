use nevermind_neu::dataloader::*;
use pleco::*;
use rand::{seq::SliceRandom, rngs::ThreadRng, Rng};
use std::cell::RefCell;

use log::info;
use rand::thread_rng;

use crate::sqlite_dataset::encode_board;

pub struct SqliteChessDataloader {
    sqlite_con: rusqlite::Connection,
    idx: RefCell<usize>,
    length: usize,
    pub do_shuffle: bool,
}

impl SqliteChessDataloader {
    pub fn new(db_path: &str) -> Self {
        let sqlite_con = rusqlite::Connection::open(db_path).expect("Failed to open sqlite3 db");

        let table_len;
        // get the table length
        {
            let query = format!("SELECT COUNT(*) FROM {}", "positions");
            let mut stmt = sqlite_con.prepare(&query).unwrap();

            let mut len_rows = stmt.query([]).unwrap();

            if let Some(row) = len_rows.next().unwrap() {
                table_len = row.get(0).unwrap();
            } else {
                panic!("Couldn't get sqlite table length");
            }
        }

        info!("Sqlite {} table length : {}", "positions", table_len);

        Self {
            sqlite_con,
            idx: RefCell::new(0),
            length: table_len,
            do_shuffle: false,
        }
    }
}

impl DataLoader for SqliteChessDataloader {
    fn next(&self) -> &LabeledEntry {
        todo!()
    }

    fn next_batch(&self, size: usize) -> MiniBatch {
        if *self.idx.borrow() + size >= self.len().unwrap() {
            info!("[SqliteChessDataloader] Going to the start...");
            let mut rng = thread_rng();

            if self.do_shuffle {
                *self.idx.borrow_mut() = rng.gen_range(0..size); // random offset
            } else {
                *self.idx.borrow_mut() = 0;
            }
        }

        let mut v = Vec::with_capacity(size);

        let query = format!(
            "SELECT * from {} LIMIT {} OFFSET {}",
            "positions",
            size,
            *self.idx.borrow()
        );

        let mut stmt = self.sqlite_con.prepare(&query).unwrap();
        let mut rows = stmt.query([]).unwrap();

        while let Some(row) = rows.next().unwrap() {
            let fen: String = row.get_unwrap(0);
            let mut eval: f32 = row.get_unwrap(1);

            // clamp (-20.0 | 20.0) eval
            if eval > 20.0 {
                eval = 20.0;
            }

            if eval < -20.0 {
                eval = -20.0;
            }

            // minmax normalize eval
            eval = (eval + 20.0) / 40.0;

            let mut board = Board::from_fen(&fen)
                .expect("[SqliteChessDataLoader] Failed to create board from fen");

            v.push(encode_board(&mut board, eval).unwrap());
        }

        *self.idx.borrow_mut() += size;

        MiniBatch::new_no_ref(v)
    }

    fn pos(&self) -> Option<usize> {
        Some(*self.idx.borrow())
    }

    fn len(&self) -> Option<usize> {
        Some(self.length)
    }
}
