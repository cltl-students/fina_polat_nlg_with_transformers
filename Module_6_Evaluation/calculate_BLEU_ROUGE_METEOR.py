import numpy as np
import sacrebleu
import glob
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')


def read_file(file):
    data_list = []
    with open(file, 'r', encoding='utf-8') as data:
        lines = data.readlines()
        for line in lines:
            line = line.lower()
            data_list.append(line)

    return data_list


def calculate_BLEU(ref_list, gen_list):
    bleu_scores = []
    for ref, gen in zip(ref_list, gen_list):
        bleu = sacrebleu.sentence_bleu(gen, [ref], smooth_method='exp').score
        bleu_scores.append(bleu)
    bleu_scores = np.array(bleu_scores).astype(np.float64)
    bleu_mean = np.mean(bleu_scores)

    return bleu_mean

def calculate_ROUGE(ref_list, gen_list):
    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    for (gen, ref) in zip(gen_list, ref_list):
        score = r_scorer.score(gen, ref)
        precision, recall, fmeasure = score['rouge1']
        rouge_scores.append(fmeasure)
    rouge_scores = np.array(rouge_scores).astype(np.float64)
    rouge_mean = np.mean(rouge_scores)

    return rouge_mean

def calculate_METEOR(ref_list, gen_list):
    meteor_scores = []
    for (gen, ref) in zip(gen_list, ref_list):
        model_output = word_tokenize(gen)
        gold_references = word_tokenize(ref)
        meteor_score = single_meteor_score(model_output, gold_references)
        meteor_scores.append(meteor_score)
    meteor_scores = np.array(meteor_scores).astype(np.float64)
    meteor_mean = np.mean(meteor_scores)

    return meteor_mean


def write_results(ref_file, gen_file, out_file):
    '''doc string'''
    ref_list = read_file(ref_file)
    gen_list = read_file(gen_file)
    mean_bleu_score = calculate_BLEU(ref_list, gen_list)
    mean_rouge_score = calculate_ROUGE(ref_list, gen_list)
    mean_meteor_score = calculate_METEOR(ref_list, gen_list)

    wo = open(out_file, 'w', encoding='utf-8')

    wo.write(f'Input file: {gen_file} \n')
    wo.write(f'Reference file: {ref_file} \n')
    wo.write(f'Results file: {out_file} \n')
    wo.write(f'SacreBLEU score: {mean_bleu_score} \n')
    wo.write(f'ROUGE score: {mean_rouge_score} \n')
    wo.write(f'METEOR score: {mean_meteor_score} \n')

    wo.close()



ref_path = '../preprocessed_data/test_reference_text.txt'
directory = 'data_ready2evaluate'

for filename in glob.iglob(f'{directory}/*'):
    if 'generated' in filename:
        gen_path = filename
        name = filename.split('_by_')
        name = name[-1]
        outfile_path = f'results/results_of_{name}'
        print(gen_path)
        print(outfile_path)
        print('\n')
        write_results(ref_path, gen_path, outfile_path)



