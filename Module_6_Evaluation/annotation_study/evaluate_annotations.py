#infile = 'sample_finetuned_T5_wiki desc.tsv'
infile = 'sample_sup_CycleGT.tsv'
print(f'Infile: {infile}')

with open(infile, 'r', encoding='utf-8') as input:
    lines = input.readlines()

hallucination_list = []
no_hall_list = []
intrinsic_list = []
extrinsic_list =[]
both_list =[]
dae_list =[]
for line in lines:
    #if line.startswith('Generation : '):
    if line.startswith('CycleGT-sup Generation :'):
        line = line.strip()
        #print(line)
        gen = line.split('\t')
        hall = gen[1]
        hallucination_list.append(hall)
        #print(hall)
        if hall == 'no':
            no_hall_list.append(1)
        else:
            intr = gen[2]
            intrinsic_list.append(intr)
            #print(intr)
            ext = gen[3]
            extrinsic_list.append(ext)
            #print(ext)
            if intr == 'yes' and ext == 'yes':
                both_list.append('yes')

    if line.startswith('Sent-level factuality probability:'):
        line = line.strip()
        dae = line.split(':')
        dae = dae[-1]
        dae = dae[:5]
        dae = dae.strip()
        dae = float(dae)
        dae_list.append(dae)
        #print(dae)


print(len(hallucination_list))
print(len(dae_list))

instances_with_hallucinations = hallucination_list.count('yes')
print(f'Number of intances that contains hallucinated generation: {instances_with_hallucinations}')
print(f'Number of intances that DO NOT contain hallucinated generation: {len(no_hall_list)}')

nr_of_intrinsic = intrinsic_list.count('yes')
print(f'Number of instances with intrinsic hallucinations: {nr_of_intrinsic}')

nr_of_extrinsic = extrinsic_list.count('yes')
print(f'Number of instances with extrinsic hallucinations: {nr_of_extrinsic}')

nr_of_both = both_list.count('yes')
print(f'Number of instances both with intrinsic and extrinsic hallucinations: {nr_of_both}')

only_intrinsic = nr_of_intrinsic - nr_of_both
print(f'Number of instances only intrinsic hallucination: {only_intrinsic}')

only_extrinsic = nr_of_extrinsic - nr_of_both
print(f'Number of instances only intrinsic hallucination: {only_extrinsic}')

dae_mean = sum(dae_list) / 100
dae_mean = round(dae_mean, 2)
print(f'Avarage DAE score of the sample: {dae_mean}')