
#path = '../generated_output/generated_by_NOTtuned_T5small.txt'
path = '../generated_output/generated_by_tuned_T5small.txt'
#path = '../generated_output/generated_by_plus_wiki_tuned_T5small.txt'
#path = '../generated_output/generated_by_plus_wiki_only_desc_tuned_T5small.txt'

#gen_path = 'data_ready2evaluate/generated_by_NOTtuned_T5small.txt'
gen_path = 'data_ready2evaluate/generated_by_tuned_T5small.txt'
#gen_path = 'data_ready2evaluate/generated_by_plus_wiki_tuned_T5small.txt'
#gen_path = 'data_ready2evaluate/generated_by_plus_wiki_only_desc_tuned_T5small.txt'


data = open(path, 'r', encoding='utf-8')
gen_file = open(gen_path, 'w', encoding='utf-8')

generated_text_list = data.readlines()
for gen in generated_text_list:
    gen = gen.replace('\n', '')
    if gen:
        gen_file.write(gen + '\n')

gen_file.close()
data.close()