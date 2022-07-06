from sklearn.metrics import f1_score
import glob

def get_main_gold_triples(gold_file):
    """ read the file, prepare the triples, return list of prepared triples """
    with open(gold_file, 'r', encoding='utf-8') as ref_triples:
        ref_triples = ref_triples.readlines()

    main_gold_triples = []
    for triples in ref_triples:
        triples = triples.strip('\n')
        triples = triples.lower()
        triples = triples.replace('|', '')
        triples = triples.replace('_', ' ')
        triples = triples.replace('  ', ' ')
        triples = triples.split(' && ')
        main_gold_triples.append(triples)

    return main_gold_triples


def get_preds_and_gold_triples(output_file):
    """ read file, extract predictions and gold, return 2 lists """
    with open(output_file, 'r', encoding='utf-8') as results:
        results = results.readlines()
        #print(len(results))

    gold_list = []
    pred_list = []

    for line in results:
        line = line.strip('\n')

        if line.startswith('=='):
            continue

        if line.startswith('Pred:'):
            pred = line
            pred_list.append(pred)

        if line.startswith('Gold:'):
            gold = line
            gold_list.append(gold)

    return pred_list, gold_list


def prepare_pred_string(pred):
    """ get the text, remove unwanted characters, return prepared text """
    pred = pred.replace('(', '')
    pred = pred.replace("',", '')
    pred = pred.replace("'", '')
    pred = pred.replace('[', '')
    pred = pred.replace(']', '')
    pred = pred.replace(',', '')
    pred = pred.replace('Pred: ', '')
    pred = pred.split(')')
    pred = list(filter(None, pred))

    return pred


def prepare_gold_string(gold):
    """ get the text, remove unwanted characters, return prepared text """
    gold = gold.replace('(', '')
    gold = gold.replace("',", '')
    gold = gold.replace("'", '')
    gold = gold.replace('[', '')
    gold = gold.replace(']', '')
    gold = gold.replace(',', '')
    gold = gold.replace('Gold: ', '')
    gold = gold.split(')')
    gold = list(filter(None, gold))

    return gold


def get_all_arrays(pred_list, gold_list, main_gold_list):
    """take 3 lists, sort and group, return 2 lists """
    all_pred_arrays = []
    all_gold_arrays = []

    for pred, gold, main_gold in zip(pred_list, gold_list, main_gold_list):

        if pred == 'Pred: []' and gold == 'Gold: []':
            pred_array = []
            gold_array = []
            for i in range(len(main_gold)):
                pred_array.append(0)
                gold_array.append(1)
            all_pred_arrays.append(pred_array)
            all_gold_arrays.append(gold_array)

        elif pred == 'Pred: []' and gold != 'Gold: []':
            pred_array = []
            gold_array = []
            for i in range(len(main_gold)):
                pred_array.append(0)
                gold_array.append(1)
            all_pred_arrays.append(pred_array)
            all_gold_arrays.append(gold_array)

        elif pred != 'Pred: []' and gold == 'Gold: []':
            pred = prepare_pred_string(pred)
            pred_array = []
            gold_array = []
            for g in main_gold:
                gold_array.append(1)
                if any(g in i for i in pred):
                    pred_array.append(1)
                else:
                    pred_array.append(0)
            all_pred_arrays.append(pred_array)
            all_gold_arrays.append(gold_array)

        elif pred != 'Pred: []' and len(gold) > 8:
            pred = prepare_pred_string(pred)
            pred_array = []
            gold_array = []
            for g in main_gold:
                gold_array.append(1)
                if any(g in i for i in pred):
                    pred_array.append(1)
                else:
                    pred_array.append(0)
            all_pred_arrays.append(pred_array)
            all_gold_arrays.append(gold_array)

        else:
            print('This should not be printed. There is something wrong here!')

    return all_pred_arrays, all_gold_arrays


def get_all_the_lists(pred_list, gold_list):
    """take 2 lists, sort and group, return 5 lists """
    gold_array_list = []
    pred_array_list = []
    no_gold_no_pred_list = []
    no_pred_list = []
    no_gold_but_pred_list = []

    for pred, gold in zip(pred_list, gold_list):
        if pred == 'Pred: []' and gold == 'Gold: []':
            no_gold_no_pred_list.append(1)

        elif pred == 'Pred: []' and gold != 'Gold: []':
            no_pred_list.append(1)

        elif pred != 'Pred: []' and gold == 'Gold: []':
            no_gold_but_pred_list.append(1)

        elif pred != 'Pred: []' and len(gold) > 8:
            pred = prepare_pred_string(pred)
            gold = prepare_gold_string(gold)
            pred_array = []
            gold_array = []
            for g in gold:
                gold_array.append(1)
                if any(g in i for i in pred):
                    pred_array.append(1)
                else:
                    pred_array.append(0)
            gold_array_list.append(gold_array)
            pred_array_list.append(pred_array)

        else:
            print('This should not be printed. There is something wrong here!')

    return gold_array_list, pred_array_list, no_gold_no_pred_list, no_pred_list, no_gold_but_pred_list


def get_f_score(gold_array_list, pred_array_list):
    """take 2 lists, get f-score for each item, return mean of the full list """
    f_score_list = []

    for x, y in zip(gold_array_list, pred_array_list):
        f_score = f1_score(x, y, average='micro')
        f_score_list.append(f_score)

    mean_f_score = sum(f_score_list) / len(f_score_list)

    #print(mean_f_score)

    return f_score_list, mean_f_score

def write_the_results(result_file, gold_file, output_file):
    """" """
    main_gold_list = get_main_gold_triples(gold_file)
    pred_list, gold_list = get_preds_and_gold_triples(output_file)
    all_pred_arrays, all_gold_arrays = get_all_arrays(pred_list, gold_list, main_gold_list)
    f_score_list, overall_f_score = get_f_score(all_gold_arrays, all_pred_arrays)
    print(f'Overall f-score: {overall_f_score}')
    gold_array_list, pred_array_list, no_gold_no_pred_list, no_pred_list, no_gold_but_pred_list = get_all_the_lists(
        pred_list, gold_list)
    partial_fscore_list, partial_fscore = get_f_score(gold_array_list, pred_array_list)
    print(f'Partial f-score: {partial_fscore}')

    number_of_instances_in_testset = len(main_gold_list)
    number_of_pred = len(pred_array_list) + len(no_gold_but_pred_list)
    number_of_no_pred = len(no_gold_no_pred_list) + len(no_pred_list)
    number_of_no_gold = len(no_gold_no_pred_list) + len(no_gold_no_pred_list)
    total = len(pred_array_list) + len(no_gold_but_pred_list) + len(no_gold_no_pred_list) + len(no_pred_list)

    with open(result_file, 'w', encoding='utf-8') as results:
        results.write(f'Evaluated file: {output_file}\n')
        results.write(f'Number of instances in the testset: {number_of_instances_in_testset}\n')
        results.write(f'Number of predictions: {number_of_pred}\n')
        results.write(f'Number of instances without prediction: {number_of_no_pred}\n')
        results.write(f'Number of instances that gold is not extracted: {number_of_no_gold}\n')
        results.write(f'Total: {total}. This must be equal to number of instance in the test set.\n')
        results.write("##############\n")
        results.write(f'F-score just for predictions: {partial_fscore}\n')
        results.write(f'Overall f-score for the full test set: {overall_f_score}\n')


##########################################################################################################################


gold_file =  '../preprocessed_data/test_input_triples.txt'
input_folder = '../Module_5_Cycle_Training/outputs'
for filename in glob.iglob(f'{input_folder}/*'):
    if 't2g_show' in filename:
        output_file = filename
        print(f'output_file: {output_file}')
        result_file_name = filename.split('\\')
        result_file_name = result_file_name[-1].replace('t2g_show', 'CycleGT')
        result_file = f'../Module_6_Evaluation/results/T2G_results_of_{result_file_name}'
        print(f'result_file: {result_file}')
        write_the_results(result_file, gold_file, output_file)

