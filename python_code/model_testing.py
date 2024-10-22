from functions import *
import torch

pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_columns', 15)

ref_directory = 'C:/Users/Thoma/OneDrive/Desktop/Farm_1/ref/'
farm1_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\data_merging\\'
model_testing_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\model_testing\\'

var_info = pd.read_csv(ref_directory+'dataset_variables.csv')
data_file_name = farm1_dir + 'full_merged_output_with_feed.csv'
model_file_name = model_testing_dir + 'Model_202409181440.pt'

model = torch.load(model_file_name, weights_only=False)
model.eval()

farm = None
startdate = '2023-04-01'
enddate = '2023-04-30'

df = pd.read_csv(data_file_name)
df,labels = event_tester(df,var_info,farm,startdate,enddate)
input_tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
pd.DataFrame(model(input_tensor).to_numpy())