

Important Files
	Viability Report.doc		Project goals & results
	Farm1\ref\dataset_variables.csv	Data dictionary
	Farm1\python_code\preprocessor.py	Preprocessing pipeline
	Farm1\python_code\main.py		Model training
	

Contents

data_merging
	Folder used for combining datasets.
	Used by folder_list_combiner() and farm1_preprocessor() functions from preprocessor.py.
feed_data
	Feed data files.  Format was never standardized.
model_testing
	Folder for model testing.  
	Contains training data (x_data/y_data), intermediate steps (all_legible, etc.).
	Also contains trained models & training logs.
Old_Files
	Older feed samples & data files.  Not currently in use.  Most not in current format.
python_code
	Contains all code used in data preprocessing & model training.
	dataloader.py	Contains function that loads pre-processed data for model training.
	functions.py	Contains all cleaning & preprocessing functions.  
			Complete list in file.
	main.py		Trains models.  Run after preprocessor.py.
	model.py	Contains models.  Contents were only used for viability testing.
	model_test..py	Used for testing completed models against a subset of data.  Incomplete.
	preprocessor.py	Converts AMS data for model use.
	unused_code.py	Functions & code chunks not used in the final version.
r_code
	linear_model_baseline.rmd
		This was used as a performance baseline for the viability test.
	model_analysis
		Contains 2 visualization functions for checking model performance.
	EDA\
		.rmd files generated during EDA.  You can safely ignore this.
raw_data
	Raw AMS data, sorted by month.
ref
	dataset_variables.csv
		This serves as a data dictionary and reference for a lot of the preprocessing functions.
		