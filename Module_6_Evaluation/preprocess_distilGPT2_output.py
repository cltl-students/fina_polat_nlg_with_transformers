

input_data = open('../generated_output/generated_by_NOTtuned_distilGPT2.txt', 'r', encoding='utf-8')
#input_data = open('../generated_output/generated_by_tuned_distilGPT2.txt', 'r', encoding='utf-8')
#input_data = open('../generated_output/generated_by_plus_wiki_tuned_distilGPT2.txt', 'r', encoding='utf-8')
#input_data = open('../generated_output/generated_by_plus_wiki_only_desc_tuned_distilGPT2.txt', 'r', encoding='utf-8')

gen_file = open('data_ready2evaluate/generated_by_NOTtuned_distilGPT2.txt', 'w', encoding='utf-8')
#gen_file = open('data_ready2evaluate/generated_by_tuned_distilGPT2.txt', 'w', encoding='utf-8')
#gen_file = open('data_ready2evaluate/generated_by_plus_wiki_tuned_distilGPT2.txt', 'w', encoding='utf-8')
#gen_file = open('data_ready2evaluate/generated_by_plus_wiki_only_desc_tuned_distilGPT2.txt', 'w', encoding='utf-8')


data_lines = input_data.readlines()
for line in data_lines:
    if line.startswith('Output:'):
        gen = line
        gen = gen.strip()
        gen = gen.replace('\n', '')
        gen = gen.split(';')
        gen = gen[-1]
        gen = gen.replace('<|endoftext|>', '')
        gen = gen.strip()

        print(gen)
        gen_file.write(gen + '\n')

gen_file.close()
