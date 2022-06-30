The data cab be found at https://gitlab.com/shimorina/webnlg-dataset

release_v3.0

if you dowload it, remember to unzip the folder and place it in 'data' folder, delete 'ru' folder (we only use English version)

In this repository, only development set can be found as an example of the dataset. 

#### run the py files in the following order:
	preprocess.py (run for training and development data)
	
	preprocess_test.py (run for test data)
#### after running these programs, you will find the following output files in 'preprocessed_data'folder:
    test_input_triples.txt
    
    test_reference_text.txt
    
    WebNLG_dev_data.tsv
    
    WebNLG_test_data.json
    
    WebNLG_test_data.tsv
    
    WebNLG_train_data.tsv
    
preprocessing is completed here. 
