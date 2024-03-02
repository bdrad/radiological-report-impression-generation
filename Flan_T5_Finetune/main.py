from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.model import SummarizationModel
from src.data_module import get_train_data_loader, get_eval_data_loader
from transformers import AutoTokenizer
import argparse


def main(config):

    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model)
    model = SummarizationModel(config)
    save_path = "UCSF_" + config.model.pretrained_model.replace("/", "_") 

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=save_path,
        verbose=True,
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=config.training.max_epochs, 
        devices=1,
        accumulate_grad_batches=config.training.accumulate_grad_batches
    )

    if 'train' in config.mode:    
        train_loader = get_train_data_loader(config, tokenizer)
        trainer.fit(model, train_loader, [
            get_eval_data_loader(config, tokenizer, phase='validation', modality='CT'),
            get_eval_data_loader(config, tokenizer, phase='validation', modality='MRI'),
            get_eval_data_loader(config, tokenizer, phase='validation', modality='US')
        ])

    if 'test' in config.mode:
        model = SummarizationModel.load_from_checkpoint(
            checkpoint_path='checkpoints/'+save_path+'.ckpt',
            config=config
        )
        test_CT_loader = get_eval_data_loader(config, tokenizer, phase='test', modality='CT')
        test_MRI_loader = get_eval_data_loader(config, tokenizer, phase='test', modality='MRI')
        test_US_loader = get_eval_data_loader(config, tokenizer, phase='test', modality='US')
        trainer.test(
            model, 
            dataloaders=[
            test_CT_loader, test_MRI_loader, test_US_loader
            ],
            verbose=False
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load configuration file for model training.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration yaml file')
    args = parser.parse_args()
    config_path = args.config
    config = OmegaConf.load(config_path)
    main(config)