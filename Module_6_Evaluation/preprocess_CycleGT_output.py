# import required module
import glob

# assign directory
directory = '../Module_5_Cycle_Training/outputs'

# iterate over files in
# just strip the lines and write to 'data_ready2evaluate' folder
for filename in glob.iglob(f'{directory}/*'):
    if 'hyp' in filename:
        outfile_path = filename.split('\\')
        outfile_path = outfile_path[-1]
        outfile_path = outfile_path.replace('hyp','CycleGT')
        outfile_path = f'data_ready2evaluate/generated_by_{outfile_path}'
        print(f'Outfile: {outfile_path}')
        infile = open(filename, 'r', encoding='utf-8')
        outfile = open(outfile_path, 'w', encoding='utf-8')
        data = infile.readlines()
        for line in data:
            line = line.strip()
            line = line.replace('\n', '')
            outfile.write(line + '\n')
