import glob
from operator import itemgetter
import random
random.seed(42)


def create_sample_from_testfile(test_file = '../preprocessed_data/WebNLG_test_data.tsv'):
    """read test file, get a random sample of 100, return the sample and the index list"""
    with open(test_file, 'r', encoding='utf-8') as tsvinput:
        test_lines = tsvinput.readlines()

    test_data_list = []
    for ind, line in enumerate(test_lines):
        instance_dict = dict()
        line = line.strip('\n')
        line = line.split('\t')
        instance_dict['Index'] = ind
        instance_dict['Nr_of_triples'] = line[1]
        instance_dict['input_triples'] = line[2].split(' && ')
        instance_dict['reference_text'] = line[3]
        test_data_list.append(instance_dict)


    sample_list = random.sample(test_data_list, 100)

    index_list = []
    for sample in sample_list:
        index = sample['Index']
        index_list.append(index)

    sample_list = sorted(sample_list, key=itemgetter('Index'))
    index_list = sorted(index_list)

    return sample_list, index_list


def get_NLG_outputs(generation_file, index_list):
    """ read NLG output file, extract the generations from the given index, return the extracted list """
    with open(generation_file, 'r', encoding='utf-8') as tsvinput:
        lines = tsvinput.readlines()
    generation_list = []
    for ind, line in enumerate(lines):
        if ind in index_list:
            instance_dict = dict()
            instance_dict['Index'] = ind
            instance_dict['Generation'] = line.strip('\n')
            generation_list.append(instance_dict)

    return generation_list


def get_T2G_predictions(T2G_file, index_list):
    """read T2G file, get instances, return list of dicts"""
    with open(T2G_file, 'r', encoding='utf-8') as results:
        results = results.read()
        results = results.split('=====================')
        results = list(filter(None, results))

    T2G_list = []
    for ind, result in enumerate(results):
        if ind in index_list:
            instance_dict = dict()
            instance_dict['Index'] = ind
            result = result.split('\n')
            result = list(filter(None, result))
            instance_dict['Prediction'] = result[0].replace('Pred: ', '')
            instance_dict['Gold'] = result[1].replace('Gold: ', '')
            T2G_list.append(instance_dict)
    T2G_list = sorted(T2G_list, key=itemgetter('Index'))

    return T2G_list


def get_DAE_predictions(dae_file, index_list):
    """Read detailed_dae_results file, get instances, return list of dicts"""

    with open(dae_file, 'r', encoding='utf-8') as results:
        results = results.read()
        results = results.split('Gold Sentence')
        results = list(filter(None, results))

    DAE_list = []
    for ind, result in enumerate(results):
        if ind in index_list:
            instance_dict = dict()
            instance_dict['Index'] = ind
            instance_dict['DAE_result'] = f'Gold sentence {result}'
            DAE_list.append(instance_dict)
    DAE_list = sorted(DAE_list, key=itemgetter('Index'))

    return DAE_list


def write_selected_outputs(generation_file, sample_list, index_list):
    """ """
    test_file = '../preprocessed_data/WebNLG_test_data.tsv'
    try:
        file_name = generation_file.split('_by_')
        file_name = file_name[-1]
        print(file_name)
        file = generation_file.split('\\')
        file = file[-1]
        print(file)
        outfile = f'selected_data_4_error_analysis/selection_of_{file_name}'
        generation_list = get_NLG_outputs(generation_file, index_list)
        print(outfile)
        outfile = open(outfile, 'w', encoding='utf-8')

        if 'NOT' in file_name: #2 files, generated_by_NOTtuned_distilGPT2.txt
            for sample, generation in zip(sample_list, generation_list):
                outfile.write(f'Beginning of the instance\n')
                for k, v in sample.items():
                    outfile.write(f'{k} : {v}\n')
                for k, v in generation.items():
                    outfile.write(f'{k} : {v}\n')
                outfile.write(f'#### End of the instance ####\n\n')

        elif 'CycleGT' in file_name: #6 files generated_by_sup_CycleGT.txt, generated_by_sup_wiki_desc_only_CycleGT.txt
            filename = file_name.split('_CycleGT')
            filename = f'{filename[0]}'
            T2G_file = f'../Module_5_Cycle_Training/outputs/{filename}_t2g_show.txt' #sup_t2g_show.txt
            DAE_file = f'DAE_results/detailed_dae_results_of_{filename}_CycleGT.txt' #detailed_dae_results_of_sup_CycleGT.txt
            T2G_list = get_T2G_predictions(T2G_file, index_list)
            DAE_list = get_DAE_predictions(DAE_file, index_list)
            for sample, generation, T2G, DAE in zip(sample_list, generation_list, T2G_list, DAE_list):
                outfile.write(f'### Beginning of the instance###\n')
                for k, v in sample.items():
                    outfile.write(f'{k} : {v}\n')
                for k, v in generation.items():
                    outfile.write(f'{k} : {v}\n')
                for k, v in T2G.items():
                    outfile.write(f'{k} : {v}\n')
                for k, v in DAE.items():
                    outfile.write(f'{k} : {v}\n')
                outfile.write(f'#### End of the instance ####\n\n')

        elif file.startswith('generated_by') and 'NOT' or 'CycleGT' not in file:
            DAE_file = f'DAE_results/detailed_dae_results_of_{file_name}' #detailed_dae_results_of_plus_wiki_tuned_distilGPT2.txt
            DAE_list = get_DAE_predictions(DAE_file, index_list)
            #print(len(DAE_list))
            for sample, generation, DAE in zip(sample_list, generation_list, DAE_list):
                outfile.write(f'### Beginning of the instance###\n')
                for k, v in sample.items():
                    outfile.write(f'{k} : {v}\n')
                for k, v in generation.items():
                    outfile.write(f'{k} : {v}\n')
                for k, v in DAE.items():
                    outfile.write(f'{k} : {v}\n')
                outfile.write(f'#### End of the instance ####\n\n')
        outfile.close()

    except:
        print('It is not a generation file')
#######################################################################################################

sample_list, index_list = create_sample_from_testfile('../preprocessed_data/WebNLG_test_data.tsv')
print(index_list)
input_folder = f'data_ready2evaluate'
for filename in glob.iglob(f'{input_folder}/*'):
    print(filename)
    write_selected_outputs(filename, sample_list, index_list)
