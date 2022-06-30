import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from torch.utils.data import Dataset, DataLoader

import logging

logging.getLogger().setLevel(logging.CRITICAL)

import warnings

warnings.filterwarnings('ignore')

from pynvml import *

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(f'Device: {device}')

def print_gpu_utilization():
    """A function to monitor memory issues """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


class WebNLG_Dataset(Dataset):
    def __init__(self):

        WebNLG_dataset_path = '../preprocessed_data/WebNLG_train_data.tsv'
        WebNLG_devdata_path = '../preprocessed_data/WebNLG_dev_data.tsv'

        self.WebNLG_list = []
        self.end_of_text_token = "<|endoftext|>"

        with open(WebNLG_dataset_path, encoding='utf-8') as tsv_file:
            train_lines = tsv_file.readlines()
            for row in train_lines:
                row = row.split('\t')
                WebNLG_str = f"Verbalize: Category: {row[0]}, Size:{row[1]}, Triples: {row[2]}; {row[-1]} {self.end_of_text_token}"
                WebNLG_str = WebNLG_str.replace('\n', '')
                self.WebNLG_list.append(WebNLG_str)

        with open(WebNLG_devdata_path, encoding='utf-8') as tsv_file:
            dev_lines = tsv_file.readlines()
            for row in dev_lines:
                row = row.split('\t')
                WebNLG_str = f"Verbalize: Category: {row[0]}, Size:{row[1]}, Triples: {row[2]}; {row[-1]} {self.end_of_text_token}"
                WebNLG_str = WebNLG_str.replace('\n', '')
                self.WebNLG_list.append(WebNLG_str)

    def __len__(self):
        return len(self.WebNLG_list)

    def __getitem__(self, item):
        return self.WebNLG_list[item]


f = open("training_details.txt", "w")

f.write(f'Device: {device}' + '\n')

### Start with uploading data ###

dataset = WebNLG_Dataset()
#dataset = dataset[:50]
print(f'Dataset lenght: {len(dataset)}')
f.write(f'Dataset lenght: {len(dataset)}' + '\n')

WebNLG_loader = DataLoader(dataset, batch_size=1)
print('Dataset loaded')

# set some training parameters:
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400

f.write(f'Batch size: {BATCH_SIZE}' + '\n')

### set the tokenizer and pre-trained GPT2 model ###

print('Loading the tokenizer and the model')

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = model.to(device)
model.resize_token_embeddings(len(tokenizer))

#to prevent memory issues
torch.cuda.empty_cache()
#to monitor memory issues
print_gpu_utilization()

# Sets the model in training mode
print('Starting to train. It will take a while ...')

model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_sample_tens = None
models_folder = 'saved_models/'
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(EPOCHS):

    print(f"EPOCH {epoch} started" + '=' * 30)
    f.write(f"EPOCH {epoch} started" + '=' * 30 + '\n')

    for idx, sample in enumerate(WebNLG_loader):

        #################### "Fit as many instance sequences into MAX_SEQ_LEN sequence as possible" logic start ####
        sample_tens = torch.tensor(tokenizer.encode(sample[0])).unsqueeze(0).to(device)
        # Skip sample from dataset if it is longer than MAX_SEQ_LEN
        if sample_tens.size()[1] > MAX_SEQ_LEN:
            continue

        # The first instance sequence in the sequence
        if not torch.is_tensor(tmp_sample_tens):
            tmp_sample_tens = sample_tens
            continue
        else:
            # The next instance does not fit in so we process the sequence and leave the last sample
            # as the start for next sequence
            if tmp_sample_tens.size()[1] + sample_tens.size()[1] > MAX_SEQ_LEN:
                work_sample_tens = tmp_sample_tens
                tmp_sample_tens = sample_tens
            else:
                # Add the training sample to sequence, continue and try to add more
                tmp_sample_tens = torch.cat([tmp_sample_tens, sample_tens[:, 1:]], dim=1)
                continue
        ################## Sequence ready, process it trough the model ##################

        outputs = model(work_sample_tens, labels=work_sample_tens)
        loss, logits = outputs[:2]
        loss.backward()
        sum_loss = sum_loss + loss.detach().data
        f.write(f'sum_loss: {sum_loss}' + '\n')

        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0
            batch_count += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0

    # Store the model after each epoch to compare the performance of them
    torch.save(model.state_dict(),
               os.path.join('saved_models/', f"gpt2_WebNLG_{epoch}.pt"))

print('Training finished and models saved.')

f.close()
