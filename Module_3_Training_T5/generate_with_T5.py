from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained('t5-small')
print('Uploaded tokenizer.')
# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the trained model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
print('Uploaded model.')

model.to(device)
print(f'Device: {device}')
model.eval()

# inference function
def generate(triple):
    input_ids = tokenizer.encode("Verbalize: {}".format(triple), return_tensors="pt").to(device)  # Batch size 1
    outputs = model.generate(input_ids, max_length=500, num_beams=5, no_repeat_ngram_size=3)
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace('<pad>', '').replace('</s>', '')
    return gen_text

test_path = '../preprocessed_data/test_input_triples.txt'
out_file = open(r'../generated_output/generated_by_NOTtuned_T5small.txt', 'w', encoding='utf-8')

generated_list = []
with open(test_path, 'r', encoding='utf-8') as test_file:
    lines = test_file.readlines()
    for triples in lines:
        print(triples)
        generated_text = generate(f'{triples}')
        print(generated_text)
        out_file.write(generated_text + '\n\n')
        generated_list.append(generated_text)

out_file.close()

data_len = len(lines)
gen_len = len(generated_text)

if data_len == gen_len:
    print('Generated sentences for all the test instances')
else:
    print(f'data_len: {data_len}, gen_len: {gen_len}')