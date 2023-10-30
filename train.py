from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

# import local dependencies
from dataset.data_module import RegressionDataModule
from model.regnet_model import RegNetRegression

config={
    "TRAIN_CSV": "train.csv",
    "VALIDATION_CSV": "val.csv",
    "DATA_DIR": "./data",
    "BATCH_SIZE": 32,
    "LR": 0.0001,
    "EPOCHS": 25,
}

if __name__=="__main__":

    data= RegressionDataModule(
        train_csv= config["TRAIN_DIR"],
        val_csv= config["VALIDATION_DIR"],
        batch_size= config["BATCH_SIZE"],
        data_dir= config["DATA_DIR"],
        val_dir= config["VAL_DIR"],
        image_dim= (350, 350),
        precropped= True,
        raw= True,
    )

    model = RegNetRegression(
        max_epochs= config["EPOCHS"],
        lr= config["LR"],
        gpus= 1,
    )
    
    checkpoint_callback=ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        every_n_epochs=1,
        save_on_train_epoch_end=True
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
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model, data)