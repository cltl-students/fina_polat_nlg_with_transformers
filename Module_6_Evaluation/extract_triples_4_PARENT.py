
data_path = '../preprocessed_data/WebNLG_test_data.tsv'
#data_path = '../preprocessed_data/WebNLG_test_data_with_Wiki.tsv'
#data_path = '../preprocessed_data/WebNLG_test_data_with_Wiki_description_only.tsv'

triple_path = 'data_ready2evaluate/triples.txt'
#triple_path = 'data_ready2evaluate/triples_with_wiki.txt'
#triple_path = 'data_ready2evaluate/triples_with_wiki_description_only.txt'


data = open(data_path, 'r', encoding='utf-8')
data_lines = data.readlines()

triple_file = open(triple_path, 'w', encoding='utf-8')

for line in data_lines:
    line = line.split('\t')
    triples = line[2]
    ### uncomment the line below for wiki triples ###
    #triples = line[2] + ' && ' + line[-1]
    triples = triples.strip()
    triples = triples.replace('\n', '')
    triples = triples.replace('|', '|||')
    triples = triples.replace(' && ', '\t')
    triple_file.write(triples+'\n')


triple_file.close()