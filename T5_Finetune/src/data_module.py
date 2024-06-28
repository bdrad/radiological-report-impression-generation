import pandas as pd
from torch.utils.data import DataLoader, Dataset

class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, config, phase='train', input_length=512, output_length=150):
        self.data = data
        self.config = config
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        input_text = "summarize: " + row['Processed Source']  
        target_text = row['Processed Impression'] 

        source = self.tokenizer.encode_plus(input_text, max_length=self.input_length, padding='max_length', truncation=True, return_tensors="pt")
        target = self.tokenizer.encode_plus(target_text, max_length=self.output_length, padding='max_length', truncation=True, return_tensors="pt")

        return {
            "modality": row['Modality'],
            "input_ids": source.input_ids.flatten(),
            "attention_mask": source.attention_mask.flatten(),
            "labels": target.input_ids.flatten(),
            "decoder_attention_mask": target.attention_mask.flatten()
        }

def get_train_data_loader(config, tokenizer):
    train_data = pd.read_csv(config.data.train_path)
    train_dataset = SummarizationDataset(
        train_data, 
        tokenizer, 
        config, 
        phase="train"
    )
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    return train_loader

def get_eval_data_loader(config, tokenizer, phase, modality):
    if phase == 'validation':
        data = pd.read_csv(config.data.validation_path)
    else:
        data = pd.read_csv(config.data.test_path)
    data = data[data['Modality'] == modality].reset_index(drop=True)
    test_dataset = SummarizationDataset(
        data,
        tokenizer, 
        config, 
        phase=phase
    )
    eval_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    return eval_loader