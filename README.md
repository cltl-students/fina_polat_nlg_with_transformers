# Master Thesis Project: Hallucinatory World of Automatic Text Generation with Transformers
Written by Fina Polat
30/06/2022

Link to the report: https://www.overleaf.com/project/6271389f4714fc28f5e5be71

This repository contains the code which is developed for the HLT Master Thesis: Hallucinatory World of Automatic Text Generation with Transformers.

The project investigates whether it is possible to reduce the amount of hallucinations in the automatically generated text by transformers.

For this, we employ the following techniques:

1) integration of additional information
2) Cycle training

The code is presented in 6 modules:

Module 1: The code in this module prepocesses the WebNLG corpus and resulting files are stored in 'preprocessed_data' directory.
Module 2: This module is developed to extract additional information about the WebNLG entities from Wikidata. Resulting files are stored in 'preprocessed_data' directory.
Module 3: In this module, we finetune pretrained language model T5-small with WebNLG, WebNLG+all additional triples from Wikidata, WebNLG+only Wikidata description. Then, we generate verbalizations with the finetuned models. Generated text is saved in 'generated_output' directory. 
Module 4: In this module, we finetune pretrained language model DistilGPT2 with WebNLG, WebNLG+all additional triples from Wikidata, WebNLG+only Wikidata description. Then, we generate verbalizations with the finetuned models. Generated text is saved in 'generated_output' directory. 
Module 5: This module experiments with CycleGT. We train CycleGT in supervised and unsupervised manner with the three versions of WebNLG corpus. All the system outputs are stored in 'outputs' folder in this module.
Module 6: In this module, we prepare the system outputs for evaluation and store the resulting files in the repository named 'data_ready2evaluate'. Then, we evaluate the generations in the following categories: traditional metrics (BLEU, ROUGE, METEOR), PARENT, and DAE. The results of traditional metrics are stored in 'results' folder. PARENT results can be found in 'PARENT_results', and DAE results are located in 'DAE_results'.

In order to run the code, please install the requirements. All modules contain another 'read_me' file. Follow the instructions there.  
For DAE evaluation, remember to start a Standford CoreNLP server.

# Code references:

https://github.com/QipengGuo/CycleGT
https://github.com/tagoyal/dae-factuality
https://github.com/tagoyal/factuality-datasets
https://github.com/google-research/language/tree/master/language/table_text_eval
https://github.com/MathewAlexander/T5_nlg/blob/main/T5_data_to_text.ipynb
https://github.com/wikimedia/pywikibot - reference for installation
https://github.com/google/sentencepiece/blob/master/python/README.md - reference for installation
https://towardsdatascience.com/teaching-gpt-2-a-sense-of-humor-fine-tuning-large-transformer-models-on-a-single-gpu-in-pytorch-59e8cec40912
