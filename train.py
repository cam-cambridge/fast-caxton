import pytorch_lightning as pl

# import local dependencies
from src.dataset.data_module import RegressionDataModule
from src.model.regnet_model import RegNetRegression

config={
    "TRAIN_CSV": "data/train.csv",
    "VAL_CSV": "data/val.csv",
    "DATA_DIR": "data/imgs",
    "BATCH_SIZE": 1,
    "LR": 0.0001,
    "EPOCHS": 10,
}

if __name__=="__main__":

    data= RegressionDataModule(
        train_csv= config["TRAIN_CSV"],
        val_csv= config["VAL_CSV"],
        batch_size= config["BATCH_SIZE"],
        data_dir= config["DATA_DIR"],
        val_dir= config["DATA_DIR"],
        image_dim= (350, 350),
        precropped= True,
        raw= True,
    )

    model = RegNetRegression(
        max_epochs= config["EPOCHS"],
        lr= config["LR"],
        gpus= 1,
    )

    trainer = pl.Trainer(
        deterministic=False,
	    devices=2,
	    accelerator= "gpu",
	    max_epochs=config["EPOCHS"],
	    strategy= "ddp_find_unused_parameters_false",
        limit_train_batches= 1.0,
        limit_val_batches= 1.0,
        precision=16,
        log_every_n_steps=2
    )
    
    trainer.fit(model, data)