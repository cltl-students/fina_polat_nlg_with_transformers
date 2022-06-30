from nltk.tokenize import sent_tokenize, TreebankWordTokenizer
# TreebankWordTokenizer recommended by DAE developers
import glob

tokenizer = TreebankWordTokenizer()

def preprocess_4_DAE(ref_path,gen_path,outfile_path ):
    ''''read references, read generation, write on a file: ref + newline + gen + blank line'''
    ref_file = open(ref_path, "r", encoding='utf-8')
    gen_file = open(gen_path, "r", encoding='utf-8')
    outfile = open(outfile_path, "w", encoding='utf-8')

    ref_lines = ref_file.readlines()
    print(f'Ref len: {len(ref_lines)}')
    gen_lines = gen_file.readlines()
    print(f'Gen len: {len(gen_lines)}')

    len_list = []
    for ref, gen in zip(ref_lines, gen_lines):
        ref = ref.lower()
        ref = ref.strip()
        ref = ref.strip('\n')
        gen = gen.lower()
        gen = gen.strip()
        gen = gen.strip('\n')

        ref_text = []
        for sent in sent_tokenize(ref):
            tokenized_sent = tokenizer.tokenize(sent)
            tokenized_sent = ' '.join(tokenized_sent)
            ref_text.append(tokenized_sent)
        ref = ' '.join(ref_text)
        #print(f'Reference text: {ref}')
        outfile.write(ref + '\n')

        gen_text = []
        for sent in sent_tokenize(gen):
            tokenized_sent = tokenizer.tokenize(sent)
            tokenized_sent = ' '.join(tokenized_sent)
            gen_text.append(tokenized_sent)

        gen_text = ' '.join(gen_text)
        number_of_tokens = len(gen_text.split(" "))
        if gen_text == '' or len(gen_text.split(" ")) < 5:
            gen = 'This is a filler text because the model did not make a proper generation.'
        if number_of_tokens > 100:
            generation = gen_text.split(' ')
            generation = generation[:100]
            gen = ' '.join(generation)
        len_list.append(len(gen.split(' ')))
        # print(f'Generated text: {gen}')
        outfile.write(gen + '\n')
        outfile.write('\n')
    ref_file.close()
    gen_file.close()
    outfile.close()
    return max(len_list)


ref_path = '../../preprocessed_data/test_reference_text.txt'
directory = '../data_ready2evaluate'

for filename in glob.iglob(f'{directory}/*'):
    if 'generated' in filename:
        gen_path = filename
        name = filename.split('_by_')
        name = name[-1]
        outfile_path = f'prepared_data/prepared_version_of_{name}'
        print(gen_path)
        print(outfile_path)
        print('\n')
        max_token_number = preprocess_4_DAE(ref_path, gen_path, outfile_path)
        print(f'Max token number: {max_token_number}')


