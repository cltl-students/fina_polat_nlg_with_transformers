from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained('t5-small')
# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the trained model

#### Model 1 ####
#model = T5ForConditionalGeneration.from_pretrained('saved_models/pytoch_model_T5WebNLG_Wiki.bin', return_dict=True,
                                                  # config='saved_models/t5-small-config.json')

#### Model 2 ####
model = T5ForConditionalGeneration.from_pretrained('saved_models/pytoch_model_T5WebNLG_Wiki_desc_only.bin', return_dict=True,
                                                   config='saved_models/t5-small-config.json')

model.to(device)
print(device)
model.eval()


# inference function
def generate(triple):
    input_ids = tokenizer.encode("Verbalize: {}".format(triple), return_tensors="pt").to(device)  # Batch size 1
    outputs = model.generate(input_ids, max_length=500, num_beams=5, no_repeat_ngram_size=3)
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace('<pad>', '').replace('</s>', '')
    return gen_text

## Choose the data file ##
#WebNLG_dataset_path = '../preprocessed_data/WebNLG_test_data_with_Wiki.tsv'
WebNLG_dataset_path = '../preprocessed_data/WebNLG_test_data_with_Wiki_description_only.tsv'
with open(WebNLG_dataset_path, 'r', encoding='utf-8') as input:
    test_lines = input.readlines()

print(f'Lenght of the test set: {len(test_lines)}')

## Choose the output file ##
#out_file = open(r'../generated_output/generated_by_plus_wiki_tuned_T5small.txt', 'w', encoding='utf-8')
out_file = open(r'../generated_output/generated_by_plus_wiki_only_desc_tuned_T5small.txt', 'w', encoding='utf-8')

for line in test_lines:
    line = line.strip('\n')
    line = line.split('\t')
    input_string = f'Category: {line[0]} Size:{str(line[1])} Triples: {line[2]} ' \
                   f'Wiki size:{str(line[4])} Wiki Triples: {line[5]}'
    print(f'Input String: {input_string }')
    generated_text = generate(input_string)
    print(f'Generated text: {generated_text}')
    out_file.write(generated_text + '\n')

out_file.close()