from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AdamW
from rouge_score import rouge_scorer
import torch
import pytorch_lightning as pl
from scipy import stats
import numpy as np
from collections import defaultdict

class SummarizationModel (pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model.pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.pretrained_model)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.outputs = {
            'CT': defaultdict(list),
            'MRI': defaultdict(list),
            'US': defaultdict(list)
        }
        self.config = config

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        return output.loss
    
    def training_step(self, batch, batch_idx):
        loss = self(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        return loss

    def evaluate_step(self, batch, batch_idx):
        labels = batch["labels"]
        modality = batch["modality"][0]
        ground_truths = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in labels]
        generated_ids = self.model.generate(batch['input_ids'], attention_mask=batch['attention_mask'],
                                            max_length=150, num_beams=4, repetition_penalty=2.5,
                                            length_penalty=1.0, early_stopping=True)
        generated_summaries = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]                
        rouge_scores = [self.scorer.score(gt, gs) for gt, gs in zip(ground_truths, generated_summaries)]
        
        rouge1_f1_scores = [scores['rouge1'].fmeasure for scores in rouge_scores]
        rouge2_f1_scores = [scores['rouge2'].fmeasure for scores in rouge_scores]
        rougeL_f1_scores = [scores['rougeL'].fmeasure for scores in rouge_scores]

        self.outputs[modality]['rouge1_f1_scores'].append(rouge1_f1_scores)
        self.outputs[modality]["rouge2_f1_scores"].append(rouge2_f1_scores)
        self.outputs[modality]["rougeL_f1_scores"].append(rougeL_f1_scores)
    
    def evaluate_epoch_end(self, phase):
        for modality in self.outputs:
            all_rouge1_f1 = np.concatenate(self.outputs[modality]["rouge1_f1_scores"])
            all_rouge2_f1 = np.concatenate(self.outputs[modality]["rouge2_f1_scores"])
            all_rougeL_f1 = np.concatenate(self.outputs[modality]["rougeL_f1_scores"])
            
            rouge1_f1_mean, rouge1_f1_lower, rouge1_f1_upper = bootstrap_confidence_interval(all_rouge1_f1)
            rouge2_f1_mean, rouge2_f1_lower, rouge2_f1_upper = bootstrap_confidence_interval(all_rouge2_f1)
            rougeL_f1_mean, rougeL_f1_lower, rougeL_f1_upper = bootstrap_confidence_interval(all_rougeL_f1)
            
            print("="*40)
            print(f"Evaluate ({phase}) | Modality ({modality})")
            print("="*40)

            print(f"ROUGE-1: {rouge1_f1_mean} ({rouge1_f1_lower}, {rouge1_f1_upper})")
            print(f"ROUGE-2: {rouge2_f1_mean} ({rouge2_f1_lower}, {rouge2_f1_upper})")
            print(f"ROUGE-L: {rougeL_f1_mean} ({rougeL_f1_lower}, {rougeL_f1_upper})")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Your validation step logic
        loss = self(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.evaluate_step(batch, batch_idx)
        return loss
    
    def on_validation_epoch_end(self):
        self.evaluate_epoch_end('Validation')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.evaluate_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.evaluate_epoch_end('Test')

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.config.training.learning_rate)
    
def bootstrap_confidence_interval(data, n_bootstrap=1000, ci=95):
    """Calculate the mean and confidence interval using bootstrapping."""
    bootstrap_means = np.random.choice(data, (n_bootstrap, len(data)), replace=True).mean(axis=1)
    lower = np.percentile(bootstrap_means, (100-ci)/2)
    upper = np.percentile(bootstrap_means, 100-(100-ci)/2)
    mean = np.mean(bootstrap_means)
    return mean, lower, upper