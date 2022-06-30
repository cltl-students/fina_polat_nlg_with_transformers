import csv
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor
import torch
from pynvml import *
import warnings

warnings.filterwarnings('ignore')


def print_gpu_utilization():
    """A function to monitor memory issues """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")

f = open("training_details.txt", "w")

# before downloading any model or tokenizer, check for the GPU availability
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

f.write(f'Device: {dev}' + '\n')

### Upload the training data ###

with open('../preprocessed_data/WebNLG_train_data.tsv', 'r', encoding='utf-8') as tsvinput:
    train_lines = tsvinput.readlines()
with open('../preprocessed_data/WebNLG_dev_data.tsv', 'r', encoding='utf-8') as tsvinput:
    dev_lines = tsvinput.readlines()

data_lines = train_lines + dev_lines

data = []
for line in data_lines:
    line = line.split('\t')
    data.append(line)

print('Data loaded.')
print(f'Lenght of the dataset: {len(data)}')

f.write(f'Lenght of the dataset: {len(data)}' + '\n')

### set the tokenizer and pre-trained T5 model ###

print('Loading the tokenizer and the model')

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)
model.to(dev)  # moving the model to device(GPU/CPU)

# Initialize the Adafactor optimizer with parameter values suggested for t5
optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
)

#to prevent memory issues
torch.cuda.empty_cache()
#to monitor memory issues
print_gpu_utilization()

#set a few parameters for training

batch_size = 4
num_of_batches = len(data_lines)/batch_size
num_of_epochs = 5
num_of_batches = int(num_of_batches)
print(f'Number of batches: {num_of_batches}')
f.write(f'Number of batches: {num_of_batches}' + '\n')

# Sets the module in training mode
print('Started to train. It will take a while ...')
model.train()

loss_per_10_steps = []
for epoch in range(1, num_of_epochs + 1):
    print('Running epoch: {}'.format(epoch))

    running_loss = 0

    for i in range(num_of_batches):
        inputbatch = []
        labelbatch = []
        batch_data = data[i * batch_size:i * batch_size + batch_size]
        #print(f'data: {batch_data}')
        for indx, row in enumerate(batch_data):
            #print(type(row))
            #print(f'row: {row}')
            input = f'Verbalize: Category: {row[0]} Size:{str(row[1])} Triples: {row[2]} </s>'
            #print(input)
            labels = str(row[-1]).strip('\n') + ' '+ '</s>'
            #print(f'target text: {labels}')
            inputbatch.append(input)
            labelbatch.append(labels)
        inputbatch = tokenizer.batch_encode_plus(inputbatch, padding=True, max_length=400, return_tensors='pt')[
            "input_ids"]
        labelbatch = tokenizer.batch_encode_plus(labelbatch, padding=True, max_length=400, return_tensors="pt")[
            "input_ids"]
        inputbatch = inputbatch.to(dev)
        labelbatch = labelbatch.to(dev)

        # clear out the gradients of all Variables
        optimizer.zero_grad()

        # Forward propogation
        outputs = model(input_ids=inputbatch, labels=labelbatch)
        loss = outputs.loss
        loss_num = loss.item()
        logits = outputs.logits
        running_loss += loss_num
        if i % 10 == 0:
            loss_per_10_steps.append(loss_num)

        # calculating the gradients
        loss.backward()

        # updating the params
        optimizer.step()

    running_loss = running_loss / int(num_of_batches)
    print('Epoch: {} , Running loss: {}'.format(epoch, running_loss))
    f.write('Epoch: {} , Running loss: {}'.format(epoch, running_loss) + '\n')
f.close()
print('Training has finished')

model_dir = 'saved_models/'
torch.save(model.state_dict(), model_dir + 'pytoch_model_T5WebNLG.bin')

print(f'Tuned model has been saved to the directory: {model_dir}')

model.eval()
input_ids = tokenizer.encode("Verbalize: Maarssen | postCode | 3605EP  </s>", return_tensors="pt")  # Batch size 1
input_ids=input_ids.to(dev)
outputs = model.generate(input_ids)
tokenizer.decode(outputs[0])
print('Test: ')
print(tokenizer.decode(outputs[0]))
print('It works :)')