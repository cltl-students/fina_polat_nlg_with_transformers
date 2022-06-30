from nltk.tokenize import wordpunct_tokenize
#since some references include multiple sentence, I prefer using wordpuct tokenizer
# instead of sentence tokenizer + word tokenizer
import simplejson as json

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
        one_t[0] = one_t[0].replace('``', '')
        one_t[0] = one_t[0].replace('@en', '')
        one_t[0] = wordpunct_tokenize(one_t[0])
        if one_t[0] not in entities:
            entities.append(one_t[0])

        one_t[1] = one_t[1].strip()

        one_t[2] = one_t[2].replace('_', ' ')
        one_t[2] = one_t[2].replace('``', '')
        one_t[2] = one_t[2].replace('@en', '')
        one_t[2] = wordpunct_tokenize(one_t[2])
        if one_t[2] not in entities:
            entities.append(one_t[2])

        processed_triples.append(one_t)

    return processed_triples, entities


def get_wiki_triples(wiki_triples):
    ''''return wiki triples and entities'''
    instance = []
    entities = []
    wiki_triples = wiki_triples.lower()
    wiki_triples = wiki_triples.split('&&')
    if len(wiki_triples) == 0:
        print(f'No wiki triple')
    for triple in wiki_triples:
        new_triple = []
        triple = triple.replace('_', ' ')
        if 'is not found in Wikidata' in triple:
            continue
        triple = triple.split(' | ')
        if len(triple) != 3:
            continue
        head = triple[0]
        head = wordpunct_tokenize(head)
        if head not in entities:
            entities.append(head)
        relation = triple[1]
        tail = triple[2]
        tail = wordpunct_tokenize(tail)
        if tail not in entities:
            entities.append(tail)
        new_triple.append(head)
        new_triple.append(relation)
        new_triple.append(tail)
        instance.append(new_triple)
    return instance, entities


def get_text(text, entity_list):
    '''take text and entity list, replace entity strings in the text, return replaced text'''
    #print(entity_list)
    text = text.lower()
    text = wordpunct_tokenize(text)
    text = ' '.join(text)
    lst = list(range(1, 56 + 1))
    number_strings = [str(i) for i in lst]
    for i, ent in enumerate(entity_list):
        ent_string = ' '.join(ent)
        #print(f'Index: {i}, Entity string: {ent_string}')
        if ent_string in text and ent_string not in number_strings:
            text = text.replace(ent_string, f' <ENT_{i}> ')

    #print(f'Replaced text: {text}')

    return text

def merge_without_duplicates(list1, list2):
    ''''concatenate two lists without duplicates, return merged list'''
    merged_list = list1 + list2
    no_duplicate_list = []
    for item in merged_list:
        if item not in no_duplicate_list:
            no_duplicate_list.append(item)

    return no_duplicate_list


data_file = '../preprocessed_data/WebNLG_dev_data_with_Wiki.tsv'
#data_file = '../preprocessed_data/WebNLG_test_data_with_Wiki.tsv'
#data_file = '../preprocessed_data/WebNLG_train_data_with_Wiki.tsv'

###############################

#data_file = '../preprocessed_data/WebNLG_dev_data_with_Wiki_description_only.tsv'
#data_file = '../preprocessed_data/WebNLG_test_data_with_Wiki_description_only.tsv'
#data_file = '../preprocessed_data/WebNLG_train_data_with_Wiki_description_only.tsv'

triple_list = []
wiki_triple_list = []
text_list = []

with open(data_file, encoding='utf-8') as tsv_file:
    data_lines = tsv_file.readlines()
    for row in data_lines:
        row = row.split('\t')
        triples = row[2]
        triple_list.append(triples)
        wiki_triples = row[-1]
        #print(wiki_triples)
        wiki_triple_list.append(wiki_triples)
        text = row[3]
        text_list.append(text)

dict_list = []
for triples, wiki_triples, text in zip(triple_list, wiki_triple_list, text_list):
    data_dict = dict()
    processed_given_triples, given_entity_list = get_triples(triples)
    wiki_processed_triples, wiki_entity_list = get_wiki_triples(wiki_triples)
    entity_list = merge_without_duplicates(given_entity_list, wiki_entity_list)
    processed_triples = merge_without_duplicates(processed_given_triples, wiki_processed_triples)
    new_text = get_text(text, entity_list)
    data_dict['relations'] = processed_triples
    data_dict['text'] = new_text
    data_dict['entities'] = entity_list
    dict_list.append(data_dict)

with open("converted_data/dev_plus_Wiki.json", "w", encoding='utf-8') as final:
   json.dump(dict_list, final, indent=1, ensure_ascii=False)

#with open("converted_data/test_plus_Wiki.json", "w", encoding='utf-8') as final:
   #json.dump(dict_list, final, indent=1, ensure_ascii=False)

#with open("converted_data/train_plus_Wiki.json", "w", encoding='utf-8') as final:
   #json.dump(dict_list, final, indent=1, ensure_ascii=False)

#########################

#with open("converted_data/dev_plus_Wiki_desc_only.json", "w", encoding='utf-8') as final:
   #json.dump(dict_list, final, indent=1, ensure_ascii=False)

#with open("converted_data/test_plus_Wiki_desc_only.json", "w", encoding='utf-8') as final:
   #json.dump(dict_list, final, indent=1, ensure_ascii=False)

#with open("converted_data/train_plus_Wiki_desc_only.json", "w", encoding='utf-8') as final:
   #json.dump(dict_list, final, indent=1, ensure_ascii=False)