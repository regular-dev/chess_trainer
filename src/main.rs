extern crate nevermind_neu;

use clap;
use clap::{App, Arg, Command};

use env_logger::Env;

pub mod create_dataset;
pub mod dataloader;
pub mod play;
pub mod sqlite_dataset;
pub mod test;
pub mod train;
pub mod util;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let matches = App::new("chess_trainer")
        .version("0.1.0")
        .author("xion")
        .about("Train and play chess with nevermind-neu")
        .subcommand_required(true)
        .subcommand(
            Command::new("gen_dataset")
                .arg(
                    Arg::new("DeskNum")
                        .short('n')
                        .long("desk_num")
                        .help("Desk number in dataset")
                        .default_value("64")
                        .require_equals(true)
                        .takes_value(true)
                        .value_parser(clap::value_parser!(usize)),
                )
                .about("LEGACY"),
        )
        .subcommand(
            Command::new("train")
                .arg(
                    Arg::new("Ocl")
                        .long("ocl")
                        .help("Use OpenCL computations")
                        .takes_value(false),
                )
                .arg(
                    Arg::new("Dataset")
                        .long("dataset")
                        .help("Path to sqlite3 database with evaluated positions")
                        .takes_value(true)
                        .required(true),
                )
                .arg(
                    Arg::new("State")
                        .long("state")
                        .help("Continue train model from some state")
                        .takes_value(true)
                        .required(false),
                )
                .arg(
                    Arg::new("Out")
                        .long("out")
                        .help("Specifies the model state filename")
                        .takes_value(true)
                        .default_value("chess_net"),
                )
                .arg(
                    Arg::new("EpochsNum")
                        .long("epochs")
                        .help("Specify number of epochs")
                        .takes_value(true)
                        .default_value("50")
                        .value_parser(clap::value_parser!(usize)),
                ),
        )
        .subcommand(
            Command::new("dataset_info")
                .arg(
                    Arg::new("Dataset")
                        .required(true)
                        .takes_value(true)
                        .help("Dataset file"),
                )
                .about("LEGACY"),
        )
        .subcommand(
            Command::new("test")
                .arg(
                    Arg::new("ModelStateWhite")
                        .long("state_white")
                        .help("Trained model for white's turn")
                        .takes_value(true)
                        .required(true),
                )
                .arg(
                    Arg::new("ModelStateBlack")
                        .long("state_black")
                        .help("Trained model for black's turn")
                        .takes_value(true)
                        .required(true),
                )
                .arg(
                    Arg::new("Fen")
                        .long("fen")
                        .help("Provide fen for test")
                        .require_equals(true)
                        .takes_value(true)
                        .required(true),
                )
                .arg(
                    Arg::new("ModelCfg")
                        .long("solver_cfg")
                        .help("Provide yaml solver cfg")
                        .require_equals(true)
                        .takes_value(true)
                        .required(false),
                )
                .arg(
                    Arg::new("Ocl")
                        .long("ocl")
                        .help("Enable OpenCL computations")
                        .takes_value(false),
                ),
        )
        .about("Test trained model on FEN")
        .subcommand(
            Command::new("play")
                .arg(
                    Arg::new("ModelStateWhite")
                        .long("state_white")
                        .help("Trained model state file")
                        .takes_value(true)
                        .required(true),
                )
                .arg(
                    Arg::new("ModelStateBlack")
                        .long("state_black")
                        .help("Trained model for black's turn")
                        .takes_value(true)
                        .required(true),
                )
                .arg(
                    Arg::new("Ocl")
                        .long("ocl")
                        .help("Use OpenCL computations")
                        .takes_value(false),
                )
                .arg(
                    Arg::new("Fen")
                        .long("fen")
                        .help("Display fen notation")
                )
                .arg(
                    Arg::new("Depth")
                        .long("depth")
                        .help("Depth for bot's search. Use values from 1 to 4.")
                        .takes_value(true)
                        .value_parser(clap::value_parser!(u16))
                        .default_value("2"),
                )
                .arg(
                    Arg::new("UnicodeDisplay")
                        .long("unicode")
                        .help("Use unicode characters to display board state")
                ),
        )
        .subcommand(
            Command::new("dataset_from_db")
                .arg(
                    Arg::new("DbPath")
                        .long("db")
                        .help("Provides db path")
                        .takes_value(true)
                        .require_equals(true)
                        .required(true),
                )
                .arg(
                    Arg::new("LimitDesk")
                        .long("limit")
                        .help("Provides limit desk")
                        .takes_value(true)
                        .default_value("100")
                        .require_equals(true)
                        .value_parser(clap::value_parser!(usize)),
                )
                .arg(
                    Arg::new("Offset")
                        .long("offset")
                        .help("Sqlite db entry offset")
                        .takes_value(true)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize)),
                ),
        )
        .about("LEGACY")
        .get_matches();

    let (cmd, args) = matches.subcommand().unwrap();

    if cmd == "train" {
        train::train_new(args, args.contains_id("Ocl"))?;
    }

    if cmd == "test" {
        if args.contains_id("Ocl") {
            test::test_ocl(args)?;
        } else {
            test::test(args)?;
        }
    }

    if cmd == "play" {
        play::play_chess(&args)?;
    }

    if cmd == "dataset_from_db" {
        sqlite_dataset::dataset_from_db(args)?;
    }

    if cmd == "dataset_info" {
        create_dataset::dataset_info(args)?;
    }

    if cmd == "train_continue" {
        todo!("impl")
        // train::train_continue(args)?;
    }

    Ok(())
}
