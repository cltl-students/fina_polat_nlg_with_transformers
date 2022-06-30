### Make sure that pywikibot is installed.

Run get_wikidata_triples.py 

It will:

	extract entities from train, dev, test files and merge all entities 
	query wikidata
	write the extracted triples to 'extracted_Wiki_triples.tsv'

The file can be found in 'preprocessed_data' folder

Run add_wiki_triples_to_WebNLG.py for train, dev and test files

It will add the extracted Wikidata triples to the last column of the data file and save the file in 'preprocessed_data' folder

The files are named as follows: WebNLG_XXXX_data_with_Wiki.tsv
