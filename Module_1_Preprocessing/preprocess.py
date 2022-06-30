import glob
import re
import xml.etree.ElementTree as ET

#files = glob.glob("../data/train/**/*.xml", recursive=True)
files = glob.glob("../data/dev/**/*.xml", recursive=True) #run the code for dev folder too.
triple_re=re.compile('(\d)triples')

entry_list = []
for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    triples_num=int(triple_re.findall(file)[0])
    for sub_root in root:
        for ss_root in sub_root:
            data_dct = dict()
            attribute_dict = ss_root.attrib
            category = attribute_dict['category']
            size = attribute_dict['size']
            data_dct['input_category'] = category
            data_dct['input_size'] = size
            triples = []
            target_text = []
            for entry in ss_root:
                if entry.tag == 'lex':
                    target_text.append(entry.text)
                for triple in entry:
                    if triple.tag == 'mtriple':
                        triple_text = triple.text
                        triple_text = triple_text.replace('"', '')
                        triples.append(triple_text)
            target_text = [i for i in target_text if i.replace('\n', '').strip() != '']
            structured_master_str = (' && ').join(triples)
            data_dct['triples'] = structured_master_str
            data_dct['text'] = target_text
            entry_list.append(data_dct)

#write a tsv file: 4 columns as follows: input_category - number of triples - triples - target text

#train_file = open('../preprocessed_data/WebNLG_train_data.tsv', 'w', encoding='utf-8')
dev_file = open('../preprocessed_data/WebNLG_dev_data.tsv', 'w', encoding='utf-8')

print('Number of entries: ' + str(len(entry_list)))
for ent in entry_list:
    input_category = ent['input_category']
    input_size = ent['input_size']
    triples = ent['triples']
    targets = ent['text']
    for target in targets:
        #train_file.write(input_category +'\t'+ input_size + '\t' + triples + '\t' + target + '\n')
        dev_file.write(input_category + '\t' + input_size + '\t' + triples + '\t' + target + '\n')
#train_file.close()
dev_file.close()