def get_entity_dictionary(wiki_file):
    '''' read file, return dict '''
    entity_dict = dict()
    with open(wiki_file, encoding='utf-8') as tsv_file:
        tsv_lines = tsv_file.readlines()
        for row in tsv_lines:
            row = row.split('\t')
            entity = row[0]
            triples = row[1]
            entity_dict[entity] = triples
    return entity_dict

def get_data_entities(data_file):
    ''' read file, return list of lists'''
    data_entities = []
    with open(data_file, encoding='utf-8') as tsv_file:
        tsv_lines = tsv_file.readlines()
        for row in tsv_lines:
            row = row.split('\t')
            triples = row[2]
            triples = triples.split('&&')
            instance = []
            for triple in triples:
                triple = triple.split('|')
                ent1 = triple[0]
                ent1 = ent1.strip()
                ent2 = triple[-1]
                ent2 = ent2.strip()
                instance.append(ent1)
                instance.append(ent2)
            data_entities.append(instance)
    return data_entities

def get_wiki_triples(entity_dict, data_entities):
    '''' return list of lists and length of the instances '''
    wiki_triples = []
    wiki_size = []
    for instance in data_entities:
        wiki_instance = set()
        for entity in instance:
            if entity in entity_dict and entity_dict[entity] != '':
                wiki_instance.add(entity_dict[entity])
            else:
                wiki_instance.add(f'{entity} is not found in Wikidata')

        wiki = [s for s in wiki_instance if s] #to clean empty strings
        wiki = (' && ').join(wiki)
        wiki = wiki.replace('\n', '')
        wiki = wiki.split(' && ')
        wiki = [s for s in wiki if s] #to clean empty strings
        #print(f'Wiki: {wiki}')
        shaped_wiki = (' && ').join(wiki)
        #print(f'Shaped: {shaped_wiki}')
        size = len(wiki)
        wiki_size.append(size)
        if size == 0:
            wiki_triples.append('not found in Wikidata')
        else:
            wiki_triples.append(shaped_wiki)
    print(f'Max size: {max(wiki_size)}')

    return wiki_triples, wiki_size

def write_new_file_with_Wiki(data_file, wiki_triples, wiki_size, out_file):
    ''' write wiki triples to tsv file '''
    out_file = open(out_file, 'w', encoding='utf-8')
    with open(data_file, 'r', encoding='utf-8') as tsvinput:
        lines = tsvinput.readlines()
        for row, wiki, size in zip(lines, wiki_triples, wiki_size):
            #print(f'Row: {row}')
            #print(f'Wiki: {wiki}')
            size = str(size).replace('\n', '')
            new_row = row + '\t' + size + '\t'+ wiki
            new_row = new_row.replace('\n', '')
            #print(f'New row: {new_row}')
            out_file.write(new_row + '\n')
    out_file.close()

    print('Done')

#wiki_file = '../preprocessed_data/extracted_Wiki_triples.tsv'
wiki_file = '../preprocessed_data/extracted_Wiki_triples_just_descriptions.tsv'

data_file = '../preprocessed_data/WebNLG_dev_data.tsv'
#data_file = '../preprocessed_data/WebNLG_test_data.tsv'
#data_file = '../preprocessed_data/WebNLG_train_data.tsv'

#out_file = '../preprocessed_data/WebNLG_dev_data_with_Wiki.tsv'
#out_file = '../preprocessed_data/WebNLG_test_data_with_Wiki.tsv'
#out_file = '../preprocessed_data/WebNLG_train_data_with_Wiki.tsv'

out_file = '../preprocessed_data/WebNLG_dev_data_with_Wiki_description_only.tsv'
#out_file = '../preprocessed_data/WebNLG_test_data_with_Wiki_description_only.tsv'
#out_file = '../preprocessed_data/WebNLG_train_data_with_Wiki_description_only.tsv'

entity_dict = get_entity_dictionary(wiki_file)
data_entities = get_data_entities(data_file)
print(f'len(data_entities): {len(data_entities)}')
wiki_triples, wiki_size = get_wiki_triples(entity_dict, data_entities)
print(f'len(wiki_triples): {len(wiki_triples)}')
print(f'len(wiki_size): {len(wiki_size)}')
write_new_file_with_Wiki(data_file, wiki_triples, wiki_size, out_file)
