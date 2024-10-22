#These are functions that are no longer used.

def min_to_hours(input_string):
    #Not Used
    minutes = int(input_string[0:input_string.find(':')])
    seconds = int(input_string[input_string.find(':')+1:])
    hours = minutes // 60
    minutes = minutes % 60
    return(str(hours)+':'+str(minutes)+':'+str(seconds))

def hours_to_days(input_string):
    #Not Used
    hours = int(input_string[0:input_string.find(':')])
    minutes = int(input_string[input_string.find(':')+1:])
    days = hours // 24
    hours = hours % 24
    seconds = '00'
    return(str(days)+':'+str(hours)+':'+str(minutes)+':'+seconds)

def monthdayyear_to_yyyymmdd(input_string):
    #Not Used
    date = input_string.split(' ')[0]
    m,d,y = date.split('/')
    return y.rjust(4,'0')+m.rjust(2,'0')+d.rjust(2,'0')

def dataset_bundler(input_file_name,output_dir,farm=None,startdate=None,enddate=None):
    """
    :param input_file_name:
    :param output_dir:
    :param farm:
    :param startdate:
    :param enddate:
    :return:
    """
    #Takes the pre-processed dataset and groups it into entries for the model.
    #This set-up groups all entries for each animal number.
    #This set-up would be used if multiple days worth of data were being fed into the model.
    df = pd.read_csv(input_file_name)

    labels = label_maker(df,response_list,output_dir)

    #Flatten & Convert to Tensor
    output = list(labels['Animal_Number'])
    max_len = 0

    for i in enumerate(labels['Animal_Number']):
        df_temp = df[df['Animal_Number'] == i[1]]
        df_temp = df_temp.drop(columns=['Animal_Number'])
        df_temp = list(df_temp.to_numpy().flatten())
        if len(df_temp) > max_len: max_len = len(df_temp)
        output[i[0]] = df_temp

    #0 padding
    for i in enumerate(output):
        output[i[0]] = output[i[0]] + [0] * (max_len - len(output[i[0]]))

    #Write copy of data
    output = pd.DataFrame(output)
    output.to_csv(os.path.join(output_dir, 'x_data.csv'))
    return(output,labels)

def dataset_bundler_2(df,output_dir):
    #Takes the pre-processed dataset and groups it into entries for the model.
    #This set-up groups all entries for each Date & animal number combination.

    index_list = df.Index.unique()
    x_data = []
    y_data = []
    max_len = 0

    for i in enumerate(index_list):
        #print(i[0])
        df_temp = df[df['Index'] == i[1]].head(10)

        y_data.append(round(pd.DataFrame.sum(df_temp.Milk_Yield)))

        df_temp = df_temp.drop(columns=['Index', 'Milk_Yield', 'Milk_Yield_Expected'])
        df_temp = list(df_temp.to_numpy().flatten())
        x_data.append(df_temp)

        #if len(df_temp) > max_len: max_len = len(df_temp)

    #0 padding
    for i in enumerate(x_data):
        x_data[i[0]] = x_data[i[0]] + [0] * (max_len - len(x_data[i[0]]))

    #Write copy of data
    x_data = pd.DataFrame(x_data)
    y_data = pd.DataFrame(y_data)
    #x_data.to_csv(os.path.join(output_dir, 'x_data.csv'))
    #y_data.to_csv(os.path.join(output_dir, 'y_data.csv'))

    return(x_data,y_data)


###test.py###
from functions import *
ind_list = ['Farm', 'Date', 'Date_Time', 'Event_Type', 'Device_Type', 'Animal_Number']
pd.set_option('display.max_columns', 32)

###New preprocessor test
print(datetime.now())
df2 = dataset_preprocessor_2(input_file_name,output_dir,farm=None,startdate=None,enddate=None)
print(datetime.now())
x_data,y_data = dataset_bundler_2(df2,output_dir)
print(datetime.now())
x_data.to_csv(os.path.join(output_dir, 'x_data.csv'))
y_data.to_csv(os.path.join(output_dir, 'y_data.csv'))
print(datetime.now())
###


#####main debugging
from dataloader import *
from model import *
import torch.utils.data as data
import random

pd.set_option('display.max_columns', 32)

output_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\dataset_test'
#Call dataloader
x_train,x_valid,y_train,y_valid = load_dataset_from_directory(output_dir, False)

#Alternate
# x_train = torch.tensor(pd.read_csv(os.path.join(output_dir,'x_train.csv')).to_numpy(),dtype=torch.float32)
# x_valid = torch.tensor(pd.read_csv(os.path.join(output_dir,'x_valid.csv')).to_numpy(),dtype=torch.float32)
# y_train = torch.tensor(pd.read_csv(os.path.join(output_dir,'y_train.csv')).to_numpy(),dtype=torch.float32)
# y_valid = torch.tensor(pd.read_csv(os.path.join(output_dir,'y_valid.csv')).to_numpy(),dtype=torch.float32)
#Get Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

#Testing
x_train=x_train[:,0:84]
x_valid=x_valid[:,0:84]
y_train=y_train[:,-1]
y_valid=y_valid[:,-1]
#logits = model(x_train[0:5])
#print('logits:',logits)
#x_batch = x_train[0:3,:]
#y_batch = y_train[0:3]

#Instantiate Model and Hyperparameters
#model = AMS_Model().to(device)
model = test_Model().to(device)
learning_rate = 0.010
loss_fn = nn.MSELoss() #MSELoss,L1Loss
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=1)

#Training
x_train, x_valid = x_train.to(device), x_valid.to(device)
y_train, y_valid = y_train.to(device), y_valid.to(device)
epoch_count, train_loss_values, valid_loss_values = [], [], []


n_epochs = 10
results = []
epoch = 0
k = 100
batch_size = 100

random.seed(1)
print('Start: ',datetime.now())
for epoch in range(n_epochs):
    model.train()
    #batchset = random.sample(range(len(loader.dataset)),batch_size)
    #for i in batchset:
    for i in range(len(loader.dataset)):
        x_batch,y_batch = loader.dataset[i:i+1]
        y_pred = model(x_batch).squeeze()
        loss = loss_fn(y_pred, y_batch)
        print(epoch, ' ', i, ' ', y_pred)
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()
        #
        i += 1
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train).squeeze()
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(x_valid).squeeze()
        test_rmse = np.sqrt(loss_fn(y_pred, y_valid))
    epoch_summary = "Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse)
    results.append(epoch_summary)
    print(datetime.now(),epoch_summary)
#####
















#Normalizer test
input_file_name = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\test\\test_out_merged.csv'
output_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\dataset_test'
df = dataset_preprocessor(input_file_name,output_dir)
labels = pd.read_csv(os.path.join(output_dir,'labels.csv'))

output = list(labels['Animal_Number'])
max_len = 0

for i in enumerate(labels['Animal_Number']):
    df_temp = df[df['Animal_Number']==i[1]]
    df_temp = df_temp.drop(columns=['Animal_Number'])
    df_temp = list(df_temp.to_numpy().flatten())
    if len(df_temp) > max_len: max_len=len(df_temp)
    output[i[0]] = df_temp

#0 pad
for i in enumerate(output):
    output[i[0]] = output[i[0]] + [0] * (max_len - len(output[i[0]]))

output = pd.DataFrame(output)
output.to_csv(os.path.join(output_dir,'x_data.csv'))

#At this point, output is a list of numpy arrays of different lengths.  We need to convert it to a tensor.  Not sure how to do that yet.

with open(os.path.join('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\dataset_test','x_data.csv'),'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerows(output)

os.path.join('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\dataset_test','x_data.csv')
x_data = pd.read_csv(os.path.join(output_dir,'x_data.csv'))
y_data = pd.read_csv(os.path.join(output_dir,'labels.csv'))

# assign directory
farm1_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\test'
f1 = os.path.join(farm1_dir,'_HISTORY EAF - Milking SHORT-1 to 3.csv')
f2 = os.path.join(farm1_dir,'_HISTORY EAF - Milking SHORT-4 to 7.csv')

#Farm 1
#Date/Time, Milking Event
df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\November_2023\\HISTORY - Milking SHORT-0 to 2.csv',sep=',')
df = farm1_milking(df, 'Farm_1','AMS_Device','Milking')
df_milk = order_indecies(df)

df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\November_2023\\HISTORY - Rumination Download-1 to 10.csv',sep=',')
df = farm1_rumination(df, 'Farm_1','AMS_Device','Rumination')
df_rumination = order_indecies(df)



df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_2\\July_2023\\export-Rumination history-07_05_2023 10_08.csv',sep=';')
df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_2\\July_2023\\7.12.2023 Lactation History.csv',sep=';')


df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\August_2023\\Device Indications-1 to 10.csv',sep=',')
df.columns = df.columns.str.replace(' ', '_')
df = df.rename(columns={'Creation_Date_': 'Date_Time',
                        'Device_Address': 'Address',
                        'Type': 'Alarm_Type',
                        'Indication_ID': 'Alarm_Indication_ID',
                        'Description': 'Alarm_Description'})
df = df.drop(columns=['Device'])

df['Date_Time'] = pd.to_datetime(df['Date_Time'])
df['Date'] = df['Date_Time'].map(lambda x: x.date())


from functions import *
ind_list = ['Farm', 'Date', 'Date_Time', 'Event_Type', 'Device_Type', 'Animal_Number','Address']
join_list = ['Farm','Date','Animal_Number']
farm1_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\test'
filename = 'milk_merged.csv'
f = os.path.join(farm1_dir, filename)
df = pd.read_csv(f, sep=',')


#df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%Y-%m-%d %H:%M:%S')
df = df.sort_values(by=['Address','Date_Time'],ascending=[True,True])

df['end_time'] = pd.to_datetime(df['Date_Time']) + pd.to_timedelta(df['Box_Time'], unit='s')
df['next_start'] = df['Date_Time'].shift(-1)
df['Downtime'] = (df['next_start'] - df['end_time']).astype('timedelta64[s]')
df['Downtime'] = df['Downtime'].map(lambda x: x if x >=0 else 0)

df = df.drop(columns=['next_start','end_time'])
df.to_csv(os.path.join(farm1_dir,'milk_merged.csv'), sep=',')

import os
import pandas as pd
from functions import *
ind_list = ['Farm', 'Date', 'Date_Time', 'Event_Type', 'Device_Type', 'Animal_Number','Address']
join_list = ['Farm','Date','Animal_Number']

#Farm 1
input_folder_list = ['C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\April_2023',
                     'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\June_2023',
                     'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\August_2023',
                     'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\November_2023']
farm1_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\test'

folder_list_combiner(input_folder_list,farm1_dir) #Moves files into correct directory
(df_milk,df_alarm) = farm1_preprocessor(farm1_dir, ind_list, join_list)

#Farm 2
farm2_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_2\\test'
df = farm2_preprocessor(farm2_dir)

#Date/Time, Milking Event
#df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_2\\3.7.2023 Farm 2 Yearly.csv',sep=';')
df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_2\\July_2023\\7.12.2023 Lactation History.csv',sep=';')
df = farm2_milking(df, 'Farm_2', 'AMS_Device', 'Milking')
df_milk = order_indecies(df)

#Date/Time, Traffic event
df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_2\\3.7.2023 Farm 2 Cow Traffic.csv',sep=';')
df = farm2_traffic(df, 'Farm_2','Traffic_Control_Device','Traffic')
df_traffic = order_indecies(df)

#Date/Time, Feeding event
#df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_2\\3.7.2023 Farm 2 Feeding Yearly.csv',sep=';')
df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_2\\July_2023\\7.12.2023 Feed History.csv',sep=';')
df = farm2_yearly_feeding(df, 'Farm_2','Traffic_Control_Device','Feeding')
df_feeding = order_indecies(df)

#Date, Rumination Data
df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_2\\July_2023\\export-Rumination history-07_05_2023 10_08.csv',sep=';')
df = farm2_rumination(df, 'Farm_2', 'AMS_Device', 'Milking')
df_rumination = order_indecies(df)

#Merge

df = df_milk.merge(df_feeding, how='outer', on=ind_list)
df = df.merge(df_traffic, how='outer', on=ind_list)
#df = df.merge(df_rumination, how='outer', on=ind_list)

#Write
df.to_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_2\\test_out_merged.csv',sep=',')

#Farm 3
farm3_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_3\\test'
df = farm3_preprocessor(farm3_dir)

#df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_3\\HISTORY - FEEDING-2_15_2023 to 3_1.csv',sep=',')
df = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_3\\HISTORY - MILKING-2_14_2023 to 2_21.csv',sep=',')



#Handle date/time
df['Date_Time'] = df['Date'] #Placeholder

df['Farm'] = farm_name
df['Device_Type'] = device_type
df['Event_Type'] = event_type

df = order_indecies(df)

df.to_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_3\\test_out.csv',sep=',')

###test2.py###
from dataloader import *
from model import *
import torch.utils.data as data
import random

pd.set_option('display.max_columns', 32)

output_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\dataset_test'
#Call dataloader
x_train,x_valid,y_train,y_valid = load_dataset_from_directory(output_dir, False)

#Get Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

#Testing
x_train=x_train[:,0:28]
x_valid=x_valid[:,0:28]
y_train=y_train[:,-2]
y_valid=y_valid[:,-2]

#Instantiate Model and Hyperparameters
#model = AMS_Model().to(device)
model = test_Model().to(device)
learning_rate = 0.010
loss_fn = nn.MSELoss() #MSELoss,L1Loss
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=1)

#Training
x_train, x_valid = x_train.to(device), x_valid.to(device)
y_train, y_valid = y_train.to(device), y_valid.to(device)
epoch_count, train_loss_values, valid_loss_values = [], [], []

n_epochs = 50
results = []
epoch = 0
k = 100
batch_size = 1000

random.seed(0)
print('Start: ',datetime.now())
for epoch in range(n_epochs):
    model.train()
    batchset = random.sample(range(len(loader.dataset)),batch_size)
    for i in batchset:
    #for i in range(len(loader.dataset)):
        x_batch,y_batch = loader.dataset[i:i+1]
        y_pred = model(x_batch).squeeze()
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()
        #print(epoch, ' ', i, ' ', y_pred, ' ', loss)
        i += 1
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train).squeeze()
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(x_valid).squeeze()
        test_rmse = np.sqrt(loss_fn(y_pred, y_valid))
    epoch_summary = "Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse)
    results.append(epoch_summary)
    print(datetime.now(),epoch_summary)
#####