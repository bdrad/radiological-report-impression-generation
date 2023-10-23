from concurrent.futures import process
import os
import pandas as pd
import tqdm
import regex as re
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
import random
import argparse
from tokenizers import AddedToken

import psutil

def print_title(title):
    print('=' * 30)
    print(title)
    print('=' * 30)

class AIGDataset(Dataset):
  def __init__(self,dataset,tokenizer,source_len,summ_len):
    self.dataset = dataset 
    self.tokenizer = tokenizer
    self.text_len = source_len
    self.summ_len = summ_len
    self.text = self.dataset['Source']
    self.summary = self.dataset['Impression']

  def __len__(self):
    return len(self.text)

  def __getitem__(self,i):
    summary = '<pad> ' + str(self.summary[i])
    text = '<pad> ' + str(self.text[i])
    source = self.tokenizer.batch_encode_plus([text],max_length=self.text_len,return_tensors='pt',pad_to_max_length=True, truncation=True) # Each source sequence is encoded and padded to max length in batches
    target = self.tokenizer.batch_encode_plus([summary],max_length=self.summ_len,return_tensors='pt',pad_to_max_length=True, truncation=True) # Each target sequence is encoded and padded to max lenght in batches

    source_ids = source['input_ids'].squeeze()
    source_masks = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_masks = target['attention_mask'].squeeze()

    return {
        'source_ids':source_ids.to(torch.long),
        'source_masks':source_masks.to(torch.long),
        'target_ids':target_ids.to(torch.long),
        'target_masks':target_masks.to(torch.long)
    	   }

def train(epoch, model,tokenizer,loader,optimizer,device, manlog=[]):
  model.train()
  progress_bar = tqdm.tqdm(loader)
  total_loss, num_iterations = 0, 0

  print('\n\n\tEpoch %s'%epoch)

  for data in progress_bar:
    y = data['target_ids'].to(device)
    y_ids = y[:,:-1].contiguous() # all ids except last one
    lm_labels = y[:,1:].clone().detach() # copy the address and detach label
    lm_labels[y[:,1:]==tokenizer.pad_token_id] = -100 # if it's padded token then assign it to -100
    source_ids = data['source_ids'].to(device)
    masks = data['source_masks'].to(device)
    outputs = model(input_ids = source_ids,attention_mask = masks,decoder_input_ids=y_ids,labels=lm_labels)
    loss  = outputs[0]
    total_loss += loss.detach().item()
    num_iterations += 1
    progress_bar.set_description('Loss: {:.2f}'.format(total_loss / num_iterations))
    optimizer.zero_grad()
    loss.backward() # optimize weights through backpropagation loss
    optimizer.step()
  
  manlog.append({'mode':'train', 'epoch':epoch, 'num_iterations':num_iterations, 'total_loss':total_loss, 'loss': total_loss / num_iterations})
    
  return manlog

def valid(epoch, model,tokenizer,loader,optimizer,device,manlog=[]):
  model.eval()
  progress_bar = tqdm.tqdm(loader)
  total_loss, num_iterations = 0, 0
  print('\n\n\tValidating')
  with torch.no_grad():
    for data in progress_bar:
      y = data['target_ids'].to(device)
      y_ids = y[:,:-1].contiguous() # all ids except last one
      lm_labels = y[:,1:].clone().detach() # copy the address and detach label
      lm_labels[y[:,1:]==tokenizer.pad_token_id] = -100 # if it's padded token then assign it to -100
      source_ids = data['source_ids'].to(device)
      masks = data['source_masks'].to(device)
      outputs = model(input_ids = source_ids,attention_mask = masks,decoder_input_ids=y_ids,labels=lm_labels)
      loss  = outputs[0]
      total_loss += loss.detach().item()
      num_iterations += 1
  manlog.append({'mode':'val', 'epoch':epoch, 'num_iterations':num_iterations, 'total_loss':total_loss, 'loss': total_loss / num_iterations})
  return (total_loss / num_iterations), manlog


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['general', 'specialized', 'finegrained'], required=True)
    parser.add_argument('--epochs', required=True)
    parser.add_argument('--batchsize', required=True)
    
    args = parser.parse_args()
    mode = args.mode
    epochs = int(args.epochs)
    batchsize = int(args.batchsize)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE:', device)
    
    
    tokenizer = T5Tokenizer.from_pretrained('google/t5-efficient-tiny-nh1')
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})


    processed_data = pd.read_csv(f'data/processed/{mode}_train_dataset.csv')

    dataset = Dataset.from_pandas(processed_data)
    train_percent = 0.9
    val_percent = 0.1

    num_train = int(len(dataset) * train_percent)
    num_val = int(len(dataset) * val_percent)

    train_dataset = dataset[:num_train]
    val_dataset = dataset[num_train:num_train + num_val]

    encoder_max_length, decoder_max_length = 400, 200

    train_dataset = AIGDataset(train_dataset,tokenizer,encoder_max_length,decoder_max_length)
    val_dataset = AIGDataset(val_dataset,tokenizer,encoder_max_length,decoder_max_length)

    train_loader = DataLoader(dataset=train_dataset,batch_size=batchsize, shuffle=True,num_workers=12)
    val_loader = DataLoader(dataset = val_dataset,batch_size=batchsize, num_workers=12)

    model = T5ForConditionalGeneration.from_pretrained('google/t5-efficient-tiny-nh1')
    model = model.to(device)
    optimizer = Adam(model.parameters(),lr=1e-4,amsgrad=True)

    max_validation_loss = float('inf')
    
    print(psutil.virtual_memory())
    
    if mode == 'general':
      path = f'models/aig_t5_tiny_weights_general'

    elif mode == 'specialized':
        pretrained_path = f'models/aig_t5_tiny_weights_general'
        path = f'models/aig_t5_tiny_weights_specialized'
        
        if os.path.exists(pretrained_path):
          model.load_state_dict(torch.load(pretrained_path))
        else:
            raise Exception('pretrained general path not found')
            
    elif mode == 'finegrained':
        pretrained_path = f'models/aig_t5_tiny_weights_specialized'
        path = f'models/aig_t5_tiny_weights_finegrained'

        if os.path.exists(pretrained_path):
          model.load_state_dict(torch.load(pretrained_path))
        else:
            raise Exception('pretrained specialized path not found')
    else:
        raise Exception('Incorrect mode')

    print_title('Training ' + str(mode))
    manlog = []
    for epoch in range(epochs):
      print(f'Epoch {epoch}')
      manlog = train(epoch,model,tokenizer,train_loader,optimizer,device,manlog)
      validation_loss, manlog = valid(epoch,model,tokenizer,val_loader,optimizer,device,manlog)
      print('***Validation Loss: %0.3f' % validation_loss)
      
      if validation_loss < max_validation_loss:
        max_loss = max_validation_loss
        torch.save(
          model.state_dict(),
          path 
        )
        
      print(psutil.virtual_memory())
      pd.DataFrame(manlog).to_csv(path + '_losses.csv')


