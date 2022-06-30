import pywikibot

site = pywikibot.Site("en", "wikipedia")


def get_entities(file_path):
    """" Take the file path, read the file, extract entities, return entity set """
    entity_set = set()
    with open(file_path, encoding='utf-8') as tsv_file:
        tsv_reader = tsv_file.readlines()
        for row in tsv_reader:
            row = row.split('\t')
            triples = row[2]
            triples = triples.split('&&')
            for triple in triples:
                triple = triple.split('|')
                ent1 = triple[0]
                ent1 = ent1.strip()
                ent2 = triple[-1]
                ent2 = ent2.strip()
                if ent1.isnumeric() == False:
                    entity_set.add(ent1)
                if ent2.isnumeric() == False:
                    entity_set.add(ent2)
    print(f'{len(entity_set)} entities extracted from {file_path}')

    return entity_set


def merge_all_entities(train_set, dev_set, test_set):
    """" Take 3 entity sets, merge them, return merged set """
    merged_set = train_set.union(dev_set, test_set)
    print(f'Files merged. Number of entities as follows. Train:{len(train_set)}, Dev:{len(dev_set)}, Test:{len(test_set)}, '
          f'Total after merge: {len(merged_set)}')
    return merged_set


def get_wikidata_triples(entity):
    """"search for the entity in wikidata, return: list of triples """
    triple_list = []
    page = pywikibot.Page(site, f'{entity}')
    try:
        item = pywikibot.ItemPage.fromPage(page)
        item_dict = item.get()
        item_describtion = item_dict['descriptions']['en']
        print(f'{entity} item_describtion: {item_describtion}')
        triple_list.append(f'{entity} | description | {item_describtion}')
        clm_dict = item_dict['claims']
        type_list = []

        try:
            type_list = clm_dict["P31"]
        except:
            print('No instanceOf relation')

        if bool(type_list) == True:
            for clm in type_list:
                clm_trgt = clm.getTarget()
                clm_trgt_dict = clm_trgt.get()
                ent_type = clm_trgt_dict['labels']['en']

                if ent_type == 'human':
                    gender_list = clm_dict["P21"]
                    for clm in gender_list:
                        clm_trgt = clm.getTarget()
                        clm_trgt_dict = clm_trgt.get()
                        gender = clm_trgt_dict['labels']['en']
                        print(f'gender: {gender}')
                        triple_list.append(f'{entity} | gender | {gender}')

    except:
        print(f'{entity} not found in wikidata.')
    print('################')
    return triple_list


train_entities = get_entities('../preprocessed_data/WebNLG_train_data.tsv')
dev_entities = get_entities('../preprocessed_data/WebNLG_dev_data.tsv')
test_entities = get_entities('../preprocessed_data/WebNLG_test_data.tsv')

all_entities = merge_all_entities(train_entities, dev_entities, test_entities)

wiki_triples_file = open('../preprocessed_data/extracted_Wiki_triples_just_descriptions.tsv', 'w', encoding='utf-8')

for entity in all_entities:
    extracted_triples = get_wikidata_triples(entity)
    extracted_triples = (' && ').join(extracted_triples)
    wiki_triples_file.write(entity + '\t'+ extracted_triples + '\n')

wiki_triples_file.close()
