#import required libraries
import xml.etree.ElementTree as ET
import json

#Unlike train and dev folders, there is one test file with references.

file = "../data/test/rdf-to-text-generation-test-data-with-refs-en.xml"
entry_list = []
tree = ET.parse(file)
root = tree.getroot()
for sub_root in root:
    for ss_root in sub_root:
        data_dct = dict()
        attribute_dict = ss_root.attrib
        category = attribute_dict['category']
        size = attribute_dict['size']
        data_dct['input_category'] = category
        data_dct['input_size'] = size
        triples=[]
        target_text=[]
        for entry in ss_root:
            if entry.tag == 'lex':
                target_text.append(entry.text)
            for triple in entry:
                if triple.tag == 'mtriple':
                    triple_text = triple.text
                    triple_text = triple_text.replace('"','')
                    triples.append(triple_text)
        target_text = [i for i in target_text if i.replace('\n', '').strip() != '']
        structured_master_str = (' && ').join(triples)
        data_dct['triples'] = structured_master_str
        data_dct['text'] = target_text
        entry_list.append(data_dct)

# 1. write the list of dicts to a json file. Text is a list of reference sentences
with open("../preprocessed_data/WebNLG_test_data.json", "w", encoding='utf-8') as final:
   json.dump(entry_list, final, indent=2)

#2. write triples and reference text to a txt file. Lines are repeated in triples file because of different verbalisations
triples_file = open('../preprocessed_data/test_input_triples.txt', 'w', encoding='utf-8')
reference_file = open('../preprocessed_data/test_reference_text.txt', 'w', encoding='utf-8')
test_file = open('../preprocessed_data/WebNLG_test_data.tsv', 'w', encoding='utf-8')

#3. write a tsv file: 4 columns as follows: input_category - number of triples - triples - target text

for ent in entry_list:
    input_category = ent['input_category']
    input_size = ent['input_size']
    triples = ent['triples']
    targets = ent['text']
    for target in targets:
        triples_file.write(triples + '\n')
        reference_file.write(target + '\n')
        test_file.write(input_category +'\t'+ input_size + '\t' + triples + '\t' + target + '\n')

triples_file.close()
reference_file.close()
test_file.close()