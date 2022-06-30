import torch
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")


##### Choose the right model #####
#### model 1 ####
#model_path = "saved_models/gpt2_WebNLG_wiki_4.pt"

#### model 2 ####
model_path = "saved_models/gpt2_WebNLG_wiki_desc_only4.pt"

config = AutoConfig.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained(model_path, config=config)

model.to(device)
model.eval()

#### Choose the output file ####
#output_file_path = f'../generated_output/generated_by_plus_wiki_tuned_distilGPT2.txt'
output_file_path = f'../generated_output/generated_by_plus_wiki_only_desc_tuned_distilGPT2.txt'

if os.path.exists(output_file_path):
    os.remove(output_file_path)

output_file = open(output_file_path, 'w', encoding='utf-8')


def generate(triples):
    WebNLG_str = f"Verbalize: {triples};"
    # print(f'{ind}, {WebNLG_str}')
    with torch.no_grad():
        generation_finished = False
        cur_ids = torch.tensor(tokenizer.encode(WebNLG_str)).unsqueeze(0).to(device)
        for i in range(400):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0, -1], dim=0)
            # Take the first(from only one in this case) batch and the last predicted embedding
            if i < 3:
                n = 20
            else:
                n = 3
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(),
                                            n=n)  # Randomly(from the topN probability distribution) select the next word
            cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(device) * next_token_id],
                                dim=1)  # Add the last word to the running sequence

            if next_token_id in tokenizer.encode('<|endoftext|>'):
                generation_finished = True
                break

        if generation_finished:
            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list)
        else:
            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list)

    output_text = output_text.replace('\n', '')

    return f'Output: {output_text}'


### Choose the correct input file ###

#WebNLG_dataset_path = '../preprocessed_data/WebNLG_test_data_with_Wiki.tsv'
WebNLG_dataset_path = '../preprocessed_data/WebNLG_test_data_with_Wiki_description_only.tsv'

with open(WebNLG_dataset_path, 'r', encoding='utf-8') as input:
    test_lines = input.readlines()

print(f'Length of the test set: {len(test_lines)}')

generated_text = []
for ind, line in enumerate(test_lines):
    print(ind)
    line = line.strip('\n')
    line = line.split('\t')
    input_text = f'Category: {line[0]}, Size:{line[1]}, Triples: {line[2]} ' \
                 f'Wiki size: {line[4]} Wiki triples: {line[5]}'
    print(f'Input text: {input_text}')
    gen_text = generate(input_text)
    gen_text = gen_text.replace('\n', '')
    generated_text.append(gen_text)
    print(f'Generated text: {gen_text}')
    output_file.write(gen_text + '\n')

print(f'Generated list len: {len(generated_text)}')

output_file.close()

