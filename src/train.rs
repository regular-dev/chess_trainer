use clap::ArgMatches;
use log::info;

use std::time::Instant;

// nevermind_neu
use nevermind_neu::dataloader::*;
use nevermind_neu::layers::*;
use nevermind_neu::orchestra::*;
use nevermind_neu::models::*;
use nevermind_neu::optimizers::*;
use nevermind_neu::util::*;

pub fn train_new(is_ocl: bool) -> Result<(), Box<dyn std::error::Error>> {
    if is_ocl {
        train_chess_ocl()?;
    } else {
        train_chess()?;
    }

    Ok(())
}

pub fn train_continue(
    _args: &ArgMatches
) -> Result<(), Box<dyn std::error::Error>> {
    todo!()
}

pub fn fill_model_with_layers(mdl: &mut Sequential) 
{
    let input_layer = InputLayer::new_box(898); // 8 * 8 * 14 + 2
    mdl.add_layer(input_layer);

    for i in 0..5 {
        let fc_layer = FcLayer::new_box(900 - i * 100, tanh_activation!());
        mdl.add_layer(fc_layer);
    }

    let euc_err_layer = EuclideanLossLayer::new_box(1, sigmoid_activation!());
    mdl.add_layer(euc_err_layer);

    mdl.compile_shapes(); // do not forget to call after layers were added
}

pub fn fill_ocl_model_with_layers(mdl: &mut SequentialOcl) 
{
    let input_layer = Box::new(InputLayerOcl::new(898)); // 8 * 8 * 14 + 2
    // TODO : maybe add constructor like InputDataLayer::new_box
    mdl.add_layer(input_layer);

    for i in 0..5 {
        let fc_layer = Box::new(FcLayerOcl::new(900 - i * 100, OclActivationFunc::Tanh));
        mdl.add_layer(fc_layer);
    }

    let mut euc_err_layer = Box::new(EuclideanLossLayerOcl::new(1));
    euc_err_layer.set_activation_function(OclActivationFunc::Sigmoid);

    mdl.add_layer(euc_err_layer);

    mdl.init_layers(); // TODO : maybe rename  same as Sequential like compile_shapes(...)
}

pub fn train_chess( ) -> Result<(), Box<dyn std::error::Error>> {
    let dataset_train = Box::new(ProtobufDataLoader::from_file("chess_train.pb")?);

    let mut mdl = Sequential::new();

    fill_model_with_layers(&mut mdl);
    mdl.set_batch_size(16);

    // Optimizer
    {
        let opt = Box::new(OptimizerAdam::new(3e-3));
        mdl.set_optim(opt);
    }

    let mut net = Orchestra::new(mdl);

    net.set_train_dataset(dataset_train);
    net.set_snap_iter(100_000);
    net.set_learn_rate_decay(0.8);
    net.set_learn_rate_decay_step(100_000);
    net.set_write_err_to_file(true);
    net.set_save_on_finish_flag(true);

    let now = Instant::now();
    net.train_for_error_or_iter(1e-4, 500_000)?;
    info!("Training finished, elapsed : {} seconds", now.elapsed().as_secs());

    Ok(())
}

pub fn train_chess_ocl( ) -> Result<(), Box<dyn std::error::Error>> {
    let dataset_train = Box::new(ProtobufDataLoader::from_file("chess_train.pb")?);

    let mut mdl = SequentialOcl::new()?;

    fill_ocl_model_with_layers(&mut mdl);
    mdl.set_batch_size(32);

    // Optimizer
    {
        let mut opt = Box::new(OptimizerOclRms::new(2e-3, mdl.queue()));
        opt.set_alpha(0.87);
        mdl.set_optim(opt);
    }

    let mut net = Orchestra::new(mdl);

    net.set_train_dataset(dataset_train);
    net.set_snap_iter(30_000);
    net.set_learn_rate_decay(0.8);
    net.set_learn_rate_decay_step(100_000);
    net.set_write_err_to_file(true);
    net.set_save_on_finish_flag(true);

    let now = Instant::now();
    net.train_epochs_or_error(25, 1e-3)?;
    info!("Training finished, elapsed : {} seconds", now.elapsed().as_secs());

    Ok(())
}

