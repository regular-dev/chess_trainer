extern crate nevermind_neu;

use clap;
use clap::{App, Arg, Command};

use env_logger::Env;

pub mod create_dataset;
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
            Command::new("gen_dataset [NOT IMPL]").arg(
                Arg::new("DeskNum")
                    .short('n')
                    .long("desk_num")
                    .help("Desk number in dataset")
                    .default_value("64")
                    .require_equals(true)
                    .takes_value(true)
                    .value_parser(clap::value_parser!(usize)),
            ),
        )
        .subcommand(
            Command::new("train").arg(
                Arg::new("Ocl")
                    .long("ocl")
                    .help("Use OpenCL computations")
                    .takes_value(false),
            ),
        )
        .subcommand(
            Command::new("dataset_info").arg(
                Arg::new("Dataset")
                    .required(true)
                    .takes_value(true)
                    .help("Dataset file"),
            ),
        )
        .subcommand(
            Command::new("test")
                .arg(
                    Arg::new("ModelState")
                        .short('s')
                        .long("state")
                        .help("Provide solver state. Weights state to start training")
                        .takes_value(true)
                        .require_equals(true)
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
                    Arg::new("ModelState")
                        .long("state")
                        .help("Trained model state file")
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
                        .takes_value(false),
                )
                .arg(
                    Arg::new("Depth")
                        .long("depth")
                        .help("Depth for bot's search")
                        .takes_value(true)
                        .value_parser(clap::value_parser!(u16))
                        .default_value("2"),
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
        .get_matches();

    let (cmd, args) = matches.subcommand().unwrap();

    if cmd == "dataset_info" {
        create_dataset::dataset_info(args)?;
    }

    if cmd == "train" {
        train::train_new(args.contains_id("Ocl"))?;
    }

    if cmd == "test" {
        if args.contains_id("Ocl") {
            test::test_ocl(args)?;
        } else {
            test::test(args)?;
        }
    }

    if cmd == "dataset_from_db" {
        sqlite_dataset::dataset_from_db(args)?;
    }

    if cmd == "play" {
        play::play_chess(&args)?;
    }

    if cmd == "train_continue" {
        todo!("impl")
        // train::train_continue(args)?;
    }

    if cmd == "gen_dataset" {
        todo!("impl")
        // let desk_num = args.get_one::<usize>("DeskNum").unwrap().clone();
        //create_dataset::create_dataset(fp, desk_num)?;
    }

    Ok(())
}
