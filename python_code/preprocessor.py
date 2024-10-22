from functions import *

###EDIT ME###
main_dir = os.getcwd().replace('python_code/','')
ref_directory = main_dir + 'ref/'
raw_data_dir = main_dir + 'raw_data/'
farm1_dir = main_dir + 'data_merging/'
output_dir = main_dir + 'model_testing/'

merged_data_filename = farm1_dir + 'full_merged_output.csv'
feed_file_name = raw_data_dir + 'feed_samples.csv'
merged_data_with_feed_filename = farm1_dir + 'full_merged_output_with_feed.csv'
###EDIT ME###

input_folder_list = [raw_data_dir + x for x in os.listdir(raw_data_dir)]

var_info = pd.read_csv(ref_directory+'dataset_variables.csv')
ind_list = pd.read_csv(ref_directory+'ind_list.csv').var_name.tolist()
join_list = pd.read_csv(ref_directory+'join_list.csv').var_name.tolist()
y_list = pd.read_csv(ref_directory+'y_list.csv').var_name.tolist()

# ind_list = ['Farm', 'Date', 'Date_Time', 'Event_Type', 'Device_Type', 'Animal_Number','Address']
# join_list = ['Farm','Date','Animal_Number']
# y_list = ['Haylage_Dry_Matter','Haylage_Lignin','Haylage_Starch', 'Haylage_CP','Haylage_aNDFom',
#           'PMR_Dry_Matter','PMR_Lignin','PMR_Starch','PMR_CP','PMR_aNDFom',
#           'Corn_Dry_Matter','Corn_Lignin','Corn_Starch','Corn_CP','Corn_aNDFom']

farm = None
startdate = None
enddate = None

# input_folder_list_farm3 = ['C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_3\\Data']
# farm3_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_3\\test'


folder_list_combiner(input_folder_list,farm1_dir) #Moves files into correct directory
farm1_preprocessor(farm1_dir, var_info) #Generates full_merged_output.csv
feed_data_preprocessor(feed_file_name,merged_data_filename,merged_data_with_feed_filename,feed_measurement_duration=4, feed_date_offset=10)
dataset_preprocessor(merged_data_with_feed_filename,output_dir,var_info,farm,startdate,enddate)