use clap::ArgMatches;
use log::info;

use std::time::Instant;

// nevermind_neu
use nevermind_neu::layers::*;
use nevermind_neu::orchestra::*;
use nevermind_neu::models::*;
use nevermind_neu::optimizers::*;
use nevermind_neu::util::*;

use crate::dataloader::SqliteChessDataloader;

pub fn train_new(args: &ArgMatches, is_ocl: bool) -> Result<(), Box<dyn std::error::Error>> {
    if is_ocl {
        train_chess_ocl(args)?;
    } else {
        train_chess(args)?;
    }

    Ok(())
}

pub fn fill_model_with_layers(mdl: &mut Sequential, add_dropout: bool)
{
    let input_layer = InputLayer::new_box(898); // 8 * 8 * 14 + 2
    mdl.add_layer(input_layer);

    for i in 0..4 {
        let mut fc_layer = FcLayer::new_box(900 - i * 100, leaky_relu_activation!());

        if add_dropout {
            fc_layer.set_dropout(0.13);
        }

        mdl.add_layer(fc_layer);
    }

    let euc_err_layer = EuclideanLossLayer::new_box(1, sigmoid_activation!());
    mdl.add_layer(euc_err_layer);

    mdl.compile_shapes(); // do not forget to call after layers were added
}

pub fn fill_ocl_model_with_layers(mdl: &mut SequentialOcl, add_dropout: bool) 
{
    let input_layer = Box::new(InputLayerOcl::new(898)); // 8 * 8 * 14 + 2
    // TODO : maybe add constructor like InputDataLayer::new_box
    mdl.add_layer(input_layer);

    for i in 0..4 {
        let mut fc_layer = Box::new(FcLayerOcl::new(900 - i * 100, OclActivationFunc::LeakyReLU));

        if add_dropout {
           fc_layer.set_dropout(0.13);
        }

        mdl.add_layer(fc_layer);
    }

    let mut euc_err_layer = Box::new(EuclideanLossLayerOcl::new(1));
    euc_err_layer.set_activation_function(OclActivationFunc::Sigmoid);

    mdl.add_layer(euc_err_layer);

    mdl.init_layers(); // TODO : maybe rename  same as Sequential like compile_shapes(...)
}

pub fn train_chess(args: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let ds_path = args.get_one::<String>("Dataset").unwrap();
    let epochs = args.get_one::<usize>("EpochsNum").unwrap();

    let mut dataset = Box::new(SqliteChessDataloader::new(ds_path.as_str()));
    dataset.do_shuffle = true;

    let mut mdl = Sequential::new();

    fill_model_with_layers(&mut mdl, true);
    mdl.set_batch_size(16);

    // Optimizer
    {
        let opt = Box::new(OptimizerAdam::new(7e-4));
        mdl.set_optim(opt);
    }

    let mut net = Orchestra::new(mdl);

    net.set_train_dataset(dataset);
    net.set_snap_iter(200_000);
    net.set_learn_rate_decay(0.7);
    net.set_learn_rate_decay_step(200_000);
    net.set_write_err_to_file(true);
    net.set_save_on_finish_flag(true);

    let now = Instant::now();
    net.train_epochs_or_error(*epochs, 1e-3)?;
    info!("Training finished, elapsed : {} seconds", now.elapsed().as_secs());

    Ok(())
}

pub fn train_chess_ocl(args: &ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let ds_path = args.get_one::<String>("Dataset").unwrap();
    let mut dataset = Box::new(SqliteChessDataloader::new(ds_path.as_str()));
    dataset.do_shuffle = true;

    let epochs = args.get_one::<usize>("EpochsNum").unwrap();

    let mut mdl = SequentialOcl::new()?;

    fill_ocl_model_with_layers(&mut mdl, true);
    mdl.set_batch_size(16);

    if let Some(state) = args.get_one::<String>("State") {
        info!("Loading model state from {}", state);
        mdl.load_state(state)?;
    }

    // Optimizer
    {
        let opt = Box::new(OptimizerOclAdam::new(7e-4, mdl.queue()));
        mdl.set_optim(opt);
    }

    let mut net = Orchestra::new(mdl);
    net.name = args.get_one::<String>("Out").unwrap().clone();

    net.set_train_dataset(dataset);
    net.set_snap_iter(200_000);
    net.set_learn_rate_decay(0.7);
    net.set_learn_rate_decay_step(200_000);
    net.set_write_err_to_file(true);
    net.set_save_on_finish_flag(true);

    let now = Instant::now();
    net.train_epochs_or_error(*epochs, 1e-3)?;
    info!("Training finished, elapsed : {} seconds", now.elapsed().as_secs());

    Ok(())
}

