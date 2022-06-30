from nltk.tokenize import wordpunct_tokenize #since some references include multiple sentence,
# I prefer using wordpuct tokenizer instead of sentence tokenizer + word tokenizer
import simplejson as json
import string

def get_triples(triples):
    '''' take a triple string, process it, return two lists '''
    processed_triples = []
    entities = []
    triples = triples.lower()
    #print(f'Triples: {triples}')
    triples = triples.split('&&')
    #print(triples)
    for one_t in triples:
        one_t = one_t.split('|')
        one_t[0] = one_t[0].replace('_', ' ')
        #one_t[0] = one_t[0].replace('"', '')
        one_t[0] = one_t[0].replace('``', '')
        #one_t[0] = one_t[0].replace("'", '')
        one_t[0] = one_t[0].replace('@en', '')
        one_t[0] = wordpunct_tokenize(one_t[0])
        if one_t[0] not in entities:
            entities.append(one_t[0])

        one_t[1] = one_t[1].strip()

        one_t[2] = one_t[2].replace('_', ' ')
        #one_t[2] = one_t[2].replace('"', '')
        one_t[2] = one_t[2].replace('``', '')
        #one_t[2] = one_t[2].replace("'", '')
        one_t[2] = one_t[2].replace('@en', '')
        one_t[2] = wordpunct_tokenize(one_t[2])
        if one_t[2] not in entities:
            entities.append(one_t[2])

        processed_triples.append(one_t)

    return processed_triples, entities


def get_text(text, entity_list):
    '''take text and entity list, replace entity strings in the text, return replaced text'''
    #print(entity_list)
    text = text.lower()
    text = wordpunct_tokenize(text)
    text = ' '.join(text)
    lst = list(range(1, 48 + 1))
    number_strings = [str(i) for i in lst]
    for i, ent in enumerate(entity_list):
        ent_string = ' '.join(ent)
        #print(f'Index: {i}, Entity string: {ent_string}')
        if ent_string in text and ent_string not in number_strings:
            text = text.replace(ent_string, f' <ENT_{i}> ')

    #print(f'Replaced text: {text}')
            #print(text)

    return text


#data_file = '../preprocessed_data/WebNLG_dev_data.tsv'
#data_file = '../preprocessed_data/WebNLG_test_data.tsv'
data_file = '../preprocessed_data/WebNLG_train_data.tsv'

triple_list = []
text_list = []
with open(data_file, encoding='utf-8') as tsv_file:
    data_lines = tsv_file.readlines()
    for row in data_lines:
        row = row.split('\t')
        triples = row[2]
        triple_list.append(triples)
        text = row[3]
        text_list.append(text)

dict_list = []
for triples, text in zip(triple_list, text_list):
    data_dict = dict()
    processed_triples, entity_list = get_triples(triples)
    new_text = get_text(text, entity_list)
    data_dict['relations'] = processed_triples
    data_dict['text'] = new_text
    data_dict['entities'] = entity_list
    dict_list.append(data_dict)

#with open("converted_data/dev.json", "w", encoding='utf-8') as final:
   #json.dump(dict_list, final, indent=1, ensure_ascii=False)

#with open("converted_data/test.json", "w", encoding='utf-8') as final:
   #json.dump(dict_list, final, indent=1, ensure_ascii=False)

with open("converted_data/train.json", "w", encoding='utf-8') as final:
   json.dump(dict_list, final, indent=1, ensure_ascii=False)