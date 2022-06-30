#adapted by Fina Polat for the thesis project
#refer to https://github.com/tagoyal/factuality-datasets/blob/main/evaluate_generated_outputs.py

from pycorenlp import StanfordCoreNLP
from train import MODEL_CLASSES
from torch.utils.data import DataLoader, SequentialSampler
import torch
import numpy as np
from train_utils import get_single_features
from sklearn.utils.extmath import softmax
import glob

def clean_phrase(phrase):
    phrase = phrase.replace('\\n', '')
    phrase = phrase.replace("\\'s", "'s")
    phrase = phrase.lower()
    return phrase


def get_tokens(sent):
    parse = nlp.annotate(sent,
                         properties={'annotators': 'tokenize', 'outputFormat': 'json', 'ssplit.isOneSentence': True})
    tokens = [(tok['word'], tok['characterOffsetBegin'], tok['characterOffsetEnd']) for tok in parse['tokens']]
    return tokens


def get_token_indices(tokens, start_idx, end_idx):
    for i, (word, s_idx, e_idx) in enumerate(tokens):
        if s_idx <= start_idx < e_idx:
            tok_start_idx = i
        if s_idx <= end_idx <= e_idx:
            tok_end_idx = i + 1
            break

    return tok_start_idx, tok_end_idx


def evaluate_summary(article_data, summary, tokenizer, model, nlp, device=torch.device("cuda", 0)):
    eval_dataset = get_single_features(summary, article_data, tokenizer, nlp)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    batch = [t for t in eval_dataloader][0]
    device = device
    batch = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        input_ids, attention, child, head = batch[0], batch[1], batch[2], batch[3]
        mask_entail, mask_cont, num_dependency, arcs = batch[4], batch[5], batch[6], batch[7]
        sent_labels = batch[8]

        inputs = {'input_ids': input_ids, 'attention': attention, 'child': child,
                  'head': head, 'mask_entail': mask_entail, 'mask_cont': mask_cont,
                  'num_dependency': num_dependency, 'sent_label': sent_labels, 'device': device}

        outputs = model(**inputs)
        dep_outputs = outputs[1].detach()
        dep_outputs = dep_outputs.squeeze(0)
        dep_outputs = dep_outputs[:num_dependency, :].cpu().numpy()

        input_full = tokenizer.convert_ids_to_tokens(input_ids[0], skip_special_tokens=False)
        input_full = ' '.join(input_full).replace('[PAD]', '').strip()

        summary = input_full.split('[SEP]')[1].strip()

        print(f'Gold Sentence:\t{input_full}')
        outfile_detailed.write(f'Gold Sentence:\t{input_full}\n')
        print(f'Generated Output:\t{summary}')
        outfile_detailed.write(f'Generated Output:\t{summary}\n')

        num_negative = 0.
        prob_factual_list = []
        for j, arc in enumerate(arcs[0]):
            arc_text = tokenizer.decode(arc)
            arc_text = arc_text.replace(tokenizer.pad_token, '').strip()

            if arc_text == '':  # for bert
                break

            softmax_probs = softmax([dep_outputs[j]])
            pred = np.argmax(softmax_probs[0])

            prob_factual = softmax_probs[0][1]
            prob_factual_list.append(prob_factual)

            if pred == 0:
                num_negative += 1
            print(f'Arc:\t{arc_text}')
            outfile_detailed.write(f'Arc:\t{arc_text}\n')
            print(f'Pred:\t{pred}')
            outfile_detailed.write(f'Pred:\t{pred}\n')
            print(f'Probs:\t0={softmax_probs[0][0]}\t1={softmax_probs[0][1]}')
            outfile_detailed.write(f'Probs:\t0={softmax_probs[0][0]}\t1={softmax_probs[0][1]}\n')

        print('\n')
        outfile_detailed.write('\n')

        mean_factuality_score = np.mean(prob_factual_list)
        print(f'Sent-level factuality probability:\t{mean_factuality_score}\n\n')
        outfile_detailed.write(f'Sent-level factuality probability:\t{mean_factuality_score}\n\n')

        if num_negative > 0:
            sent_factuality_score = 0
            print(f'Sent-level prediction:\t0\n\n')
            outfile_detailed.write(f'Sent-level prediction:\t1\n\n')
        else:
            sent_factuality_score = 1
            print(f'Sent-level prediction:\t1\n\n')
            outfile_detailed.write(f'Sent-level prediction:\t1\n\n')

    return sent_factuality_score, mean_factuality_score


input_folder = 'prepared_data'

for filename in glob.iglob(f'{input_folder}/*'):

    if 'NOT' in filename:
        continue
    else:
        input_file = filename
        print(input_file)
    name = filename.split('_of_')
    name = name[-1]
    output_file = f'../DAE_results/dae_results_of_{name}'
    print(output_file)
    output_file2 = f'../DAE_results/detailed_dae_results_of_{name}'
    print(output_file2)

    if __name__ == '__main__':
        model_dir = 'DAE_model'
        model_type = 'electra_dae'

        # set up parser
        nlp = StanfordCoreNLP('http://localhost:9000')

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        tokenizer = tokenizer_class.from_pretrained(model_dir)
        model = model_class.from_pretrained(model_dir)
        device = torch.device("cuda", 0)
        device = device

        model.to(device)
        model.eval()

        input_file = open(input_file, encoding='utf-8')
        input_data = [line.strip() for line in input_file.readlines()]
        outfile_detailed = open(output_file2, "a", encoding='utf-8')
        outfile = open(output_file, "a", encoding='utf-8')

        sent_factuality_score_list = []
        exact_factuality_score_list = []

        for idx in range(0, len(input_data), 3):
            article_text = input_data[idx]
            summary = input_data[idx + 1]
            print(article_text)
            print(summary)
            factuality_score, mean_factuality_score = evaluate_summary(article_text, summary, tokenizer, model, nlp)
            sent_factuality_score_list.append(factuality_score)
            exact_factuality_score_list.append(mean_factuality_score)

        print(
            f'According to the model, the number of factual sentences is {sent_factuality_score_list.count(1)} out of '
            f'{len(sent_factuality_score_list)} sentences')
        outfile.write(
            f'According to the model, the number of factual sentences is {sent_factuality_score_list.count(1)} out of '
            f'{len(sent_factuality_score_list)} sentences' + '\n')
        print(
            f'Sentence level factuality percentage is {sent_factuality_score_list.count(1) / len(sent_factuality_score_list)}')
        outfile.write(
            f'Sentence level factuality percentage is {sent_factuality_score_list.count(1) / len(sent_factuality_score_list)}' + '\n')
        print(f'Exact factuality score for the full evaluation set is {np.mean(exact_factuality_score_list)}')
        outfile.write(f'Exact factuality score for the full evaluation set is {np.mean(exact_factuality_score_list)}')

        input_file.close()
        outfile.close()
        outfile_detailed.close()


