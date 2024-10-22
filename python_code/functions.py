import os
import time as time
from datetime import date,datetime,timedelta
import pandas as pd
import numpy as np
import shutil
import csv

"""
minsec_to_seconds       Time formatter.
hourmin_to_seconds      Time formatter
hourminsec_to_seconds   Time formatter
folder_list_combiner    Creates a copy of files from multiple folders in target folder.
na_filler               Fills NA values in a dataframe.
var_normalizer          Normalizes a set of numeric variables in a dataframe.
var_standardizer        Standardizes a set of numeric variables in a dataframe.
type_assignment         
order_indecies
augment_milking_data    Calculates time since AMS device was last used, and adds that time in second to each event.

farm1_preprocessor() Dependencies
farm1_milking           Initial process for a single milking file from Farm 1.  Format like 'HISTORY - Milking SHORT-67 to 71.csv'
farm1_rumination        Initial process for a single rumination file from Farm 1.  Format like 'HISTORY - Rumination Download-5-9.csv'
farm1_device            Initial process for a single device/alarm file from Farm 1.  Format like 'HISTORY Device Indications-10 to 30.csv'

farm2_preprocessor() Dependencies
farm2_milking
farm2_rumination
farm2_yearly_feeding
farm2_traffic

farm3_preprocessor() Dependencies
farm3_feed
farm3_milking

Preprocessors
farm1_preprocessor      Preprocesses and merged a folder containing Farm 1 data.
farm2_preprocessor      Preprocesses and merged a folder containing Farm 1 data.
farm3_preprocessor      Preprocesses and merged a folder containing Farm 1 data.

feed_data_preprocessor  Combines feed data with output from farm1_preprocessor.
dataset_preprocessor    Prepares combined feed/merged farm data for model training.

label_maker             Creates label file for model training.
dataset_bundler_lstm    Combines all events from dataset_preprocessor() into a single row by date/cow, with 0 padding.
daily_summary_bundler   Combines all events from dataset_preprocessor() as a daily summary by date/cow.
results_logger          Logs results from model training.

event_tester            Streamlined version of dataset_preprocessor meant for model testing.
event_tester_bundler    Streamlined version of dataset_bundler functions for event_tester().
"""











def minsec_to_seconds(input_string):
    #Converts 'M:SS' format to seconds.  Not optimized.
    minutes = int(input_string[0:input_string.find(':')])
    seconds = int(input_string[input_string.find(':')+1:])
    return seconds + (minutes*60)

def hourmin_to_seconds(input_string):
    #Converts 'H:MM' format to seconds.  Not optimized.
    hours,minutes = input_string.split(':')
    hours,minutes = int(hours),int(minutes)
    return (minutes*60)+(hours*3600)

def hourminsec_to_seconds(input_string):
    #This was used for some weird date/time formatting for Farm 2.
    if isinstance(input_string,str):
        if(input_string.find('d')>-1):
            d = input_string.split('d')[0].strip()
            input_string = input_string.split('d')[1].strip()
        else:
            d=0
        h,m,s = input_string.split(':')
        return int(d) * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
    else:
        return(-1)

def folder_list_combiner(input_folder_list,output_folder):
    """
    :param input_folder_list: List of folders to take files from
    :param output_folder: Folder to copy to.
    :return: None.
    """
    for i in input_folder_list:
        for fname in os.listdir(i):
            shutil.copy2(os.path.join(i,fname), output_folder)

def na_filler(df,method,vars):
    """
    Used for filling NAs in a dataframe.
    :param df: A dataframe.
    :param method: Either numerical or string.
    :param vars: List of variables to apply method to.
    :return: Dataframe with NA values in vars[] removed.
    """
    if method==0 | method=='0':
        for i in vars:
            df[i] = df[i].fillna(0)
    if method==1 | method=='mode':
        for i in vars:
            df[i] = df[i].fillna(df[i].mode().values[0])
    if method==2 | method=='mean':
        for i in vars:
            df[i] = df[i].fillna(df[i].mean().values[0])
    if method==3 | method=='median':
        for i in vars:
            df[i] = df[i].fillna(df[i].median())
    return(df)

def var_normalizer(df,vars):
    for i in vars:
        if df[i].max() != 0:
            df[i] = (df[i] - df[i].min())  / (df[i].max() - df[i].min())
    return(df)
def var_standardizer(df,vars):
    for i in vars:
        if df[i].std() != 0:
            df[i] = (df[i] - df[i].mean())  / df[i].std()
    return(df)

def type_assignment(df_data,df_vars):
    """
    :param df_data: A dataframe.
    :param df_vars: A dataframe from dataset_variables.csv.
    :return: df_data with variables ordered and type-corrected, based on dataset_variables.csv
    """
    df_data = pd.read_csv('C:\\Users\\Thoma\\OneDrive\\Desktop\\Farm_1\\test\\full_merged_output.csv')
    df_vars = pd.read_csv(ref_directory + 'ind_list.csv')
    var_names,var_type = df_vars.var_name,df_vars.type
    dict = {var_names[i]: var_type[i] for i in range(len(var_names))}
    df_data = df_data.astype(dict)

    ind_list = df_vars.var_names.tolist()
    cols = list(set(cols) - set(ind_list))  # Remaining columns
    cols = ind_list + cols

    return(df_data)

def order_indecies(df, ind_list):
    #Older version of type_assignment
    # Re-order columns
    cols = df.columns.tolist()
    ind_list = ['Farm', 'Date', 'Date_Time', 'Event_Type', 'Device_Type', 'Animal_Number']
    cols = list(set(cols) - set(ind_list))  # Remaining columns
    cols = ind_list + cols
    df = df.astype({'Farm': 'object',
               'Date': 'datetime64[ns]',
               'Date_Time': 'datetime64[ns]',
               'Event_Type': 'object',
               'Device_Type': 'object',
               'Animal_Number': 'int64'})
    return(df[cols])

def augment_milking_data(df):
    #Adds downtime (seconds) to milk_merged data
    #Downtime is based on Date_Time and Box_Time
    #Date_Time is only accurate to the minute, so negative values are common
        #*Example: Cow attempts to enter, fails, another cow enters within 60 seconds = negative difference
    #~99% of Downtimes are below 900 seconds (15 minutes)
    df = df.sort_values(by=['Address', 'Date_Time'], ascending=[True, True])

    df['end_time'] = pd.to_datetime(df['Date_Time']) + pd.to_timedelta(df['Box_Time'], unit='s')
    df['next_start'] = df['Date_Time'].shift(-1)
    df['Downtime'] = (df['next_start'] - df['end_time']).astype('timedelta64[s]')
    df['Downtime'] = df['Downtime'].map(lambda x: x if x >= 0 else 0)

    df = df.drop(columns=['next_start', 'end_time'])
    return(df)

# #######
# Farm 1
# #######
def farm1_milking(df, farm_name='default_farm', device_type = 'default_device', event_type = 'Milking'):
    """
    This functions formats milking data from (type) AMS systems.
    :param df: A dataframe generated from a milking data file.
    :param device_type: Category of AMS device.
    :param farm_name: Farm data was generated from.
    :param event_type: Type of event each row represents.  Leave as default.
    :return: A formatted dataframe of milking data.
    """
    #Fix column names
    df.columns = df.columns.str.replace(' ', '_')
    df = df.rename(columns={'Interval': 'Last_Milking_Interval'})
#    df = df.rename(columns={'Interval': 'Last_Milking_Interval',
#                            'Refusal_Type': 'Visit_Result'})

    #Fix column formatting
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    df['Box_Time'] = df['Box_Time'].map(lambda x: minsec_to_seconds(x) if not pd.isna(x) else 0)
    df['Milk_Time'] = df['Milk_Time'].map(lambda x:minsec_to_seconds(x) if not pd.isna(x) else 0)
    df['Failure'] = df['Failure'].map(lambda x:True if x=='x' else False)
    #These two do not currently occur in the Farm 3 sample:
    df['Last_Milking_Interval'] = df['Last_Milking_Interval'].map(lambda x:hourmin_to_seconds(x) if not pd.isna(x) else 0)
    df['Current_Lactation'] = df['Current_Lactation'].map(lambda x:True if x=='x' else False)

    #Remove duplicates & non-useful information
    df = df.drop_duplicates()

    #Add index columns
    df['Date'] = df['Date_Time'].map(lambda x: x.date())
    df['Farm'] = farm_name
    df['Device_Type'] = device_type
    df['Event_Type'] = event_type
    return (df)

def farm1_rumination(df, farm_name='default_farm', device_type = 'default_device', event_type = 'Rumination'):
    """
    This functions formats rumination data from (type) AMS systems.
    :param df: A dataframe generated from a rumination data file.
    :param farm_name: Farm data was generated from.
    :param device_type: Category of AMS device.
    :param event_type: Type of event each row represents.  Leave as default.
    :return: A formatted dataframe of rumination data.
    """
    #Column Fix
    superheader_list = df.columns.to_list()
    superheader_check_list = ['Total Intake','Total','Rest Feed']

    for i in superheader_check_list:
        if i not in superheader_list:
            print('Farm 1 Rumination file missing ' + i)
            superheader_list.append(i)
            superheader_list.append('')
            df[i + ' Robot Feed 1'] = ''
            df[i + ' Robot Feed 2'] = ''

    feed_column_index = [superheader_list.index(superheader_check_list[0]),
                         superheader_list.index(superheader_check_list[1]),
                         superheader_list.index(superheader_check_list[2])]

    col_list = df.iloc[0].to_list()
    col_list[feed_column_index[0]] = 'Total Intake Robot Feed 1'
    col_list[feed_column_index[0]+1] = 'Total Intake Robot Feed 2'
    col_list[feed_column_index[1]] = 'Total Robot Feed 1'
    col_list[feed_column_index[1]+1] = 'Total Robot Feed 2'
    col_list[feed_column_index[2]] = 'Rest Feed Robot Feed 1'
    col_list[feed_column_index[2]+1] = 'Rest Feed Robot Feed 2'

    df.columns = col_list
    df.columns = df.columns.str.replace(' ', '_')
    df = df.drop(0)

    #TODO: Set data types, handle NAs
    # df.astype({'Animal_Number': 'int64',
    #            'Rumination_Minutes': 'int64',
    #            'Total_Eating_Minutes': 'int64',
    #            'Chews_Per_Bolus': 'int64',
    #            'Lactation_Days': 'int64',
    #            'Lactation_Number': 'int64',
    #            'Total_Intake_Robot_Feed_1': 'float64',
    #            'Total_Intake_Robot_Feed_2': 'float64',
    #            'Total_Robot_Feed_1': 'float64',
    #            'Total_Robot_Feed_2': 'float64',
    #            'Date': 'datetime64[ns]',
    #            'Rest_Feed_Robot_Feed_1': 'float64',
    #            'Rest_Feed_Robot_Feed_2': 'float64'}).dtypes

    df = df.astype({'Date': 'datetime64[ns]'})

    #Filter rows/columns with no/non-useful information
    df = df.dropna(subset=['Total_Eating_Minutes', 'Rumination_Minutes'],how='all')
    df = df.drop(columns=['Lactation_Days','Lactation_Number'])
    df = df.drop_duplicates()

    #Columns used after combining datasets
    df['Date_Time'] = df['Date'] #Placeholder
    df['Farm'] = farm_name
    df['Device_Type'] = device_type
    df['Event_Type'] = event_type
    return (df)

def farm1_device(df, farm_name='default_farm', device_type = 'default_device', event_type = 'Device'):
    """
    This functions formats rumination data from (type) AMS systems.
    :param df: A dataframe generated from a rumination data file.
    :param farm_name: Farm data was generated from.
    :param device_type: Category of AMS device.
    :param event_type: Type of event each row represents.  Leave as default.
    :return: A formatted dataframe of rumination data.
    """
    #Fix column names
    df.columns = df.columns.str.replace(' ', '_')
    df = df.rename(columns={'Creation_Date_': 'Date_Time',
                            'Device_Address': 'Address',
                            'Type': 'Alarm_Type',
                            'Indication_ID': 'Alarm_Indication_ID',
                            'Description': 'Alarm_Description'})
    df = df.drop(columns=['Device'])

    #Fix Formatting
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])

    #Remove extraneous information & Duplicates
    df = df.drop_duplicates()

    #Add joining columns
    df['Date'] = df['Date_Time'].map(lambda x: x.date())
    df['Animal_Number'] = 0
    df['Farm'] = farm_name
    df['Device_Type'] = device_type
    df['Event_Type'] = event_type

    return(df)
# #######
# Farm 2
# #######
def farm2_milking(df, farm_name='default_farm', device_type = 'default_device', event_type = 'Milking'):
    """
    This function is used for TODO devices.  Farm dropped out of study, so is likely out of date.
    :param df: A dataframe generated from a rumination data file.
    :param farm_name: Farm data was generated from.
    :param device_type: Category of AMS device.
    :param event_type: Type of event each row represents.  Leave as default.
    :return: A formatted dataframe of milking data.
    """
    # Remove spaces from column names
    df.columns = df.columns.str.replace(' ', '_')

    # Convert time to duration in seconds
    df['Last_Milking_Interval'] = df['Last_Milking_Interval'].fillna('00:00:-1')
    df['Last_Milking_Interval'] = df['Last_Milking_Interval'].map(lambda x: hourminsec_to_seconds(x))


    df['Begin_Time'] = df['Begin_Time'].map(lambda x: datetime.strptime(x, "%m/%d/%Y %I:%M %p"))

    df['End_Time'] = df['End_Time'].map(lambda x: datetime.strptime(x, "%I:%M %p"))
    df['End_Time'] = (df['End_Time'] - df['Begin_Time'])
    df['End_Time'] = df['End_Time'].map(lambda x: x.seconds)


    df = df.rename(columns={"Begin_Time": "Date_Time",
                            "End_Time": "Duration",
                            "Yield":"Milk_Yield",
                            "Expected_Yield":"Milk_Yield_Expected",
                            "Days_In_Milk":"Lactation_Days"})

    # Teat-specific milking variables
    # All, LF,LR,LR,RR
    teat_var_names = ['Kickoff', 'Not_Milked_Teats', 'Incomplete']
    teat_list = ['LF', 'RF', 'LR', 'RR']

    for i in teat_var_names:
        df[i].fillna('')
        df[i + '_All'] = df[i] == 'All'
        for j in teat_list:
            df[j + '_' + i] = df[i].str.contains(j)
            df[j + '_' + i] = df[j + '_' + i] + df[i + '_All']
            df[j + '_' + i] = df[j + '_' + i].astype('bool')

        df = df.drop(columns=[i + '_All'])
    df = df.drop(columns=teat_var_names + ['Incomplete_Milkings'])

    df['Farm'] = farm_name
    df['Device_Type'] = device_type
    df['Event_Type'] = event_type
    return (df)

def farm2_rumination(df):
    """
    Farm 2 rumination data was never provided.
    :param df: farm 2 rumination data
    :return: farm 2 rumination data formatted
    """
    # Remove spaces from column names
    df.columns = df.columns.str.replace(' ', '_')
    # Missing Date
    return (df)

def farm2_yearly_feeding(df, farm_name='default_farm', device_type = 'default_device', event_type = 'Feeding'):
    #This was meant to handle feeding data from farm 2, specifically, what cows were moving where.  Farm 2 was dropped from the study, so it was not finished.
    #
    # Remove spaces from column names
    df.columns = df.columns.str.replace(' ', '_')
    # Format dates and times
    df['Event_Time'] = df['Event_Time'].map(lambda x: datetime.strptime(x, "%m/%d/%Y %I:%M %p"))
    df['Event_Time.1'] = df['Event_Time'].map(lambda x: x.date())
    df = df.rename(columns={'Event_Time': 'Date_Time', 'Event_Time.1': 'Date'})

    # Drop unwated variables
    #df = df.drop(columns=['Feed_Name'])

    df['Farm'] = farm_name
    df['Device_Type'] = device_type
    df['Event_Type'] = event_type
    return (df)

def farm2_traffic(df, farm_name='default_farm', device_type = 'default_device', event_type = 'Traffic'):
    #This was meant to handle "traffic" data from farm 2, specifically, what cows were moving where.  Farm 2 was dropped from the study, so it was not finished.
    # Remove spaces from column names
    df.columns = df.columns.str.replace('/', '_')
    df.columns = df.columns.str.replace(' ', '_')
    # Format dates and times
    df['Date_Time'] = df['Date_Time'].map(lambda x: datetime.strptime(x, "%m/%d/%Y %I:%M:%S %p"))
    df['Date'] = df['Date_Time'].map(lambda x: x.date())
    df['Time_in_Area_(hh:mm)'] = df['Time_in_Area_(hh:mm)'].map(lambda x:hourmin_to_seconds(x))

    df = df.rename(columns={'Time_in_Area_(hh:mm)': 'Time_In_Area'})
    # Drop unwated variables
    #df = df.drop(columns=['Feed_Name'])

    df['Farm'] = farm_name
    df['Device_Type'] = device_type
    df['Event_Type'] = event_type
    return (df)


def farm3_feed(df, farm_name='default_farm', device_type = 'default_device', event_type = 'Feeding'):
# General Indicators or duplicated in milk
# Milkings - Duplicated in milking
# Lactation_Number - Duplicated in milking
# Lactation_Days - Duplicated in Milking
# Day_Production - What is this?
# Failures - Visit Result
# Failures_Avg - Visit Result

# What are these?
# Lactose_indication (All blank?)
# Protein_indication
# Fat_indication
# Concentrate___100_lb_Milk
# Intake_Total vs Total_Programmed
# Total programmed = expected
# Intake total = total consumption?
# Day_production_Dev

# Column handling
# Rest_Feed
    # Keep
# Rest_Feed_1
    # Duplicate of rest_feed, drop
# Rumination_Minutes
    # Keep

    df.columns = df.columns.str.replace('.', '')
    df.columns = df.columns.str.replace('/', '_')
    df.columns = df.columns.str.replace(' ', '_')

    # Exclude summary rows
    df = df[df['Animal_Number'] != 'AVG']
    df = df[df['Animal_Number'] != 'SUM']

    #Column handling
    #df['Event_Time'] = df['Event_Time'].map(lambda x: datetime.strptime(x, "%m/%d/%Y %I:%M %p"))
    #df['Event_Time.1'] = df['Event_Time'].map(lambda x: x.date())
    df = df.drop(['Rest_Feed_1'],axis=1) #Duplicate of Rest_Feed
    df['Date_Time'] = df['Date']  # Placeholder



    df['Farm'] = farm_name
    df['Device_Type'] = device_type
    df['Event_Type'] = event_type
    return(df)

def farm3_milking(df, farm_name='default_farm', device_type = 'default_device', event_type = 'Milking'):
    #Fix column names
    df.columns = df.columns.str.replace(' ', '_')
    df = df.rename(columns={'Interval': 'Last_Milking_Interval',
                            'Refusal_Type': 'Visit_Result'})

    #Fix column formatting
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    df['Box_Time'] = df['Box_Time'].map(lambda x: minsec_to_seconds(x) if not pd.isna(x) else 0)
    df['Milk_Time'] = df['Milk_Time'].map(lambda x:minsec_to_seconds(x) if not pd.isna(x) else 0)
    #df['Last_Milking_Interval'] = df['Last_Milking_Interval'].map(lambda x:hourmin_to_seconds(x) if not pd.isna(x) else 0)
    #df['Current_Lactation'] = df['Current_Lactation'].map(lambda x:True if x=='x' else False)
    df['Failure'] = df['Failure'].map(lambda x:True if x=='x' else False)

    #Add index columns
    df['Date'] = df['Date_Time'].map(lambda x: x.date())
    df['Farm'] = farm_name
    df['Device_Type'] = device_type
    df['Event_Type'] = event_type
    return (df)

def farm1_preprocessor(farm1_dir, var_info):
    """
    This combines and formats the milking, rumination, and device data from farm 1.
    Must run folder_list_combiner() first on farm1_dir before running this.
    :param farm1_dir: The directory farm 1's data files are stored in.
    :param var_info: Dataframe from dataset_variables.csv
    :return:
    """

    ind_list = var_info[var_info.analysis_category=='index'].var_name.tolist()
    join_list = var_info[var_info.join==1].var_name.tolist()

    df_milk = None
    df_rumination = None
    df_alarm = None
    df_merged = None

#Iterate through each file, categorize based on substring in filename, run appropriate processing function for file.
    for filename in os.listdir(farm1_dir):
        df = None
        f = os.path.join(farm1_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if 'merged' in f.lower():
                print(f+' skipped.')
            else:
                print(f+' processed.')
                if 'milk' in f.lower():
                    df = pd.read_csv(f, sep=',')
                    df = farm1_milking(df, 'Farm_1', 'AMS_Device', 'Milking')
                    df['filename'] = f #TESTLINE
                    df = order_indecies(df,ind_list)
                    if df_milk is not None:
                        #df_milk['Date'] = pd.to_datetime(df_milk['Date_Time'])
                        #df_milk = pd.concat([df_milk, df], axis=0,ignore_index=True).reset_index(inplace=True,drop=True)
                        #df_milk =  df_milk.merge(df, how='outer', on=ind_list)
                        df_milk = pd.concat([df_milk, df])

                        df_milk = order_indecies(df_milk,ind_list)
                    else:
                        df_milk = df
                if 'rumination' in f.lower():
                    df = pd.read_csv(f, sep=',')
                    df = farm1_rumination(df, 'Farm_1', 'AMS_Device', 'Rumination')
                    df = order_indecies(df,ind_list)
                    if df_rumination is not None:
                        #df_rumination =  df_rumination.merge(df, how='outer', on=ind_list)
                        df_rumination = pd.concat([df_rumination, df])

                        df_rumination = order_indecies(df_rumination,ind_list)
                    else:
                        df_rumination = df
                if 'device' in f.lower():
                    df = pd.read_csv(f, sep=',')
                    df = farm1_device(df, 'Farm_1', 'AMS_Device', 'Alarm')
                    df = order_indecies(df,ind_list)
                    if df_alarm is not None:
                        df_alarm = pd.concat([df_alarm, df])
                        df_alarm = order_indecies(df_alarm,ind_list)
                    else:
                        df_alarm = df

#Individual merged files
    if df_milk is not None:
        df_milk = df_milk.drop_duplicates()
        df_milk = augment_milking_data(df_milk)
        df_milk.to_csv(os.path.join(farm1_dir, 'milk_merged.csv'), sep=',')
    if df_rumination is not None:
        df_rumination = df_rumination.drop_duplicates()
        df_rumination.to_csv(os.path.join(farm1_dir, 'rumination_merged.csv'), sep=',')
    if df_alarm is not None:
        df_alarm = df_alarm.drop_duplicates()
        df_alarm.to_csv(os.path.join(farm1_dir, 'alarm_merged.csv'), sep=',')

#Clean up rows missing downtime
    df_milk = df_milk.sort_values(by=['Downtime'],ascending=False)
    df_milk = df_milk.drop_duplicates(subset=df_milk.columns.difference(['Downtime']),keep='first')
    df_milk = df_milk.sort_values(by=['Date_Time'])

#Combining Event (Date/Time) Data
    if df_milk is not None and df_rumination is not None:
        print('Merging.')
        df_merged = df_milk.merge(df_rumination, how='inner', on=join_list,suffixes=[None,'_delete_me_please'])
        df_merged = df_merged[df_merged.columns.drop(list(df_merged.filter(regex='delete_me_please')))]
        if df_alarm is not None:
            df_merged = df_merged.merge(df_alarm, how='outer', on=ind_list, suffixes=[None, '_delete_me_please'])
#Write Combined
    if df_merged is not None:
        df_merged = df_merged.drop_duplicates(keep='first')
        df_merged.to_csv(os.path.join(farm1_dir,'full_merged_output.csv'), sep=',')
        print('Files merged successfully.')
    return(df_milk,df_alarm)

def farm2_preprocessor(farm2_dir):
    """
    :param farm2_dir: Directory containing farm2 files.
    :return:
    """
    ind_list = ['Farm', 'Date', 'Date_Time', 'Event_Type', 'Device_Type', 'Animal_Number']
    join_list = ['Farm','Date','Animal_Number']
    df_milk = None
    df_rumination = None
    df_merged = None

    for filename in os.listdir(farm2_dir):
        df = None
        f = os.path.join(farm2_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if 'merged' in f.lower():
                print(f+' skipped.')
            else:
                print(f+' processed.')
                if 'lactation' in f.lower():
                    df = pd.read_csv(f, sep=';')
                    df = farm2_milking(df, 'Farm_2', 'AMS_Device', 'Milking')
                    df = order_indecies(df,ind_list)
                    if df_milk is not None:
                        df_milk = pd.concat([df_milk, df])
                        df_milk = order_indecies(df_milk,ind_list)
                    else:
                        df_milk = df
                if 'rumination' in f.lower():
                    df = pd.read_csv(f, sep=';')
                    df = farm2_rumination(df, 'Farm_2', 'AMS_Device', 'Rumination')
                    df = order_indecies(df,ind_list)
                    if df_rumination is not None:
                        #df_rumination =  df_rumination.merge(df, how='outer', on=ind_list)
                        df_rumination = pd.concat([df_rumination, df])

                        df_rumination = order_indecies(df_rumination,ind_list)
                    else:
                        df_rumination = df

    print(df_milk)
    print(df_rumination)
#Individuals
    if df_milk is not None:
        df_milk.to_csv(os.path.join(farm2_dir, 'milk_merged.csv'), sep=',')
    if df_rumination is not None:
        df_rumination.to_csv(os.path.join(farm2_dir, 'rumination_merged.csv'), sep=',')
#Combining Event (Date/Time) Data
    if df_milk is not None and df_rumination is not None:
        print('Merging.')
        df_merged = df_milk.merge(df_rumination, how='inner', on=join_list,suffixes=[None,'_delete_me_please'])
        df_merged = df_merged[df_merged.columns.drop(list(df_merged.filter(regex='delete_me_please')))]
#Write Combined
    if df_merged is not None:
        df_merged = df_merged.drop_duplicates(keep='first')
        df_merged.to_csv(os.path.join(farm2_dir,'full_merged_output.csv'), sep=',')
        print('Files merged successfully.')
    return(df)

def farm3_preprocessor(farm3_dir):
    """
    :param farm3_dir: Directory containing farm3 files.
    :return:
    """
    ind_list = ['Farm', 'Date', 'Date_Time', 'Event_Type', 'Device_Type', 'Animal_Number']
    join_list = ['Farm','Date','Animal_Number']
    df_milk = None
    df_rumination = None
    df_merged = None

    for filename in os.listdir(farm3_dir):
        df = None
        f = os.path.join(farm3_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if 'merged' in f.lower():
                print(f+' skipped.')
            else:
                print(f+' processed.')
                if 'milk' in f.lower():
                    df = pd.read_csv(f, sep=',')
                    df = farm3_milking(df, 'Farm_3', 'AMS_Device', 'Milking')
                    df = order_indecies(df,ind_list)
                    if df_milk is not None:
                        #df_milk['Date'] = pd.to_datetime(df_milk['Date_Time'])
                        #df_milk = pd.concat([df_milk, df], axis=0,ignore_index=True).reset_index(inplace=True,drop=True)
                        #df_milk =  df_milk.merge(df, how='outer', on=ind_list)
                        df_milk = pd.concat([df_milk, df])

                        df_milk = order_indecies(df_milk,ind_list)
                    else:
                        df_milk = df
                if 'rumination' in f.lower():
                    df = pd.read_csv(f, sep=',')
                    df = farm3_rumination(df, 'Farm_3', 'AMS_Device', 'Rumination')
                    df = order_indecies(df,ind_list)
                    if df_rumination is not None:
                        #df_rumination =  df_rumination.merge(df, how='outer', on=ind_list)
                        df_rumination = pd.concat([df_rumination, df])

                        df_rumination = order_indecies(df_rumination,ind_list)
                    else:
                        df_rumination = df

    print(df_milk)
    print(df_rumination)
#Individuals
    if df_milk is not None:
        df_milk.to_csv(os.path.join(farm3_dir, 'milk_merged.csv'), sep=',')
    if df_rumination is not None:
        df_rumination.to_csv(os.path.join(farm3_dir, 'rumination_merged.csv'), sep=',')
#Combining Event (Date/Time) Data
    if df_milk is not None and df_rumination is not None:
        print('Merging.')
        df_merged = df_milk.merge(df_rumination, how='inner', on=join_list,suffixes=[None,'_delete_me_please'])
        df_merged = df_merged[df_merged.columns.drop(list(df_merged.filter(regex='delete_me_please')))]
#Write Combined
    if df_merged is not None:
        df_merged = df_merged.drop_duplicates(keep='first')
        df_merged.to_csv(os.path.join(farm1_dir,'full_merged_output.csv'), sep=',')
        print('Files merged successfully.')
    return(df)

def feed_data_preprocessor(feed_file_name,merged_data_filename,merged_data_with_feed_filename,feed_measurement_duration=7, feed_date_offset=7):
    """

    #Example: If feed_measurement_duration is set to 4 and feed_date_offset is set to 10, a feed sample taken January 15 would be applied to January 25,26,27,28.  So, AMS data from January 26 would try to predict feed composition from ~6-14 days prior.

    :param feed_file_name: File containing feed samples & results.
    :param merged_data_filename: File containing merged milking, alarm, & rumination data.
    :param merged_data_with_feed_filename: Output file location.
    :param feed_measurement_duration: Number of days to apply each feed result for.
    :param feed_date_offset: Number of days after a feed sample is taken we expect changes in milking/cow behavior.
    :return: Dataframe with feed values attached to each row.
    """
    feed_df = pd.read_csv(feed_file_name)
    feed_df = feed_df[["Date_Sampled","aNDFom","CP","Starch","Lignin","Dry_Matter","Product_Type"]]
    feed_df = feed_df.sort_values(by=['Date_Sampled'])

    data_df = pd.read_csv(input_file_name)

    # feed_df = feed_df[["Date_Processed","Description1","aNDFom","CP","Starch","Lignin","Product_Type"]]
    #
    # #Parse description to get farm,date,and type
    # feed_df["Description1"] = feed_df["Description1"].str.replace("-"," ")
    # feed_df["Description1"] = feed_df["Description1"].str.split(" ")
    # feed_df = pd.concat([feed_df, pd.DataFrame(feed_df.Description1.tolist(), index=feed_df.index, columns=['Farm','Date','Type1','Type2','Type3'])], axis = 1)
    #
    # feed_df = feed_df[feed_df.Farm != 'LP'] #Filter out Farm 2
    # feed_df=feed_df.drop(["Type1","Type2","Type3"],axis=1)
    #
    #
    # feed_df["Probable_Year"] = pd.DatetimeIndex(feed_df['Date_Processed']).year
    # feed_df["Date_Sampled"] = feed_df["Date"].astype(str) + "/" + feed_df["Probable_Year"].astype(str)
    #
    # #Here, we want to reduce the year by 1 if Date_Sampled is after Date_Processed
    # #feed_df["Probable_Year_2"] = feed_df["Probable_Year"][feed_df[feed_df["Date_Sampled"] > feed_df["Date_Processed"]].index] - 1
    #
    # for index, row in feed_df.iterrows():
    #     if row.Date_Sampled > row.Date_Processed:
    #         feed_df["Probable_Year"][index] = int(row.Probable_Year - 1)
    #
    # feed_df["Date_Sampled"] = feed_df["Date"].astype(str) + "/" + feed_df["Probable_Year"].astype(str)
    #
    # feed_df = feed_df[["Farm","Date_Sampled","Product_Type","aNDFom","CP","Starch","Lignin"]]

# Product_Type defines the feed type:
# 2 - Corn Silage, 7 - Ryelage, 9 - PMR, 1D - Haylage/Alfalfa Silage, 1E - Italian Rye
    typedict = {"2":"Corn",
                "9":"PMR",
                "1D":"Haylage"}
    response_var_list = ["aNDFom","CP","Starch","Lignin","Dry_Matter"]

    for i in typedict.values():
        for j in response_var_list:
            data_df.insert(2, i + "_" + j, np.nan)

    #Here, we loop through each feed type.  We create a temporary feed dataframe of only that feed type, then loop through all sample dates.  For each sample date, we look for matching rows in the data_df.  Then, for each response variable, we populate
    #the corresponding columns from the temp_df.

    datelist = []
    for key, value in zip(typedict.keys(),typedict.values()):
        temp_df = feed_df[feed_df["Product_Type"]==key]

        for index1, row1 in temp_df.iterrows():
            sample_date = row1.Date_Sampled
            datetime.strftime(datetime.strptime(sample_date, "%Y-%m-%d") + timedelta(days=feed_date_offset),"%Y-%m-%d")
            for i in range(feed_measurement_duration):
                sample_date = datetime.strftime(datetime.strptime(sample_date, "%Y-%m-%d") + timedelta(days=1), "%Y-%m-%d")
                datelist.append(sample_date)
                for j in response_var_list:
                    data_df[value + "_" + j][data_df.Date==sample_date] = row1[j]

    data_df = data_df[data_df.Date.isin(datelist)]
    data_df.to_csv(data_file_name)

    return(data_df)
#def training_recorder(output_dir,model_name,model,):

def dataset_preprocessor(data_file_name,output_dir,var_info,farm=None,startdate=None,enddate=None):
    """
    :param data_file_name: Output file from feed_data_preprocessor().
    :param output_dir: Directory to output files to.
    :param var_info: Dataframe of dataset_variables.csv
    :param farm: This would filter to only results from a specific farm.  Currently unused.
    :param startdate: This would filter to only results from a specific date range.  Currently unused.
    :param enddate: This would filter to only results from a specific date range.  Currently unused.
    :return:

    #Takes the "Analysis version" of AMS data (full_merged_output.csv) and converts it for model use.
    #Current output consists of 15 variables.  Each animal is filed seperately, with rows sorted by ascending Date_Time.
    #Automated Milking System (AMS) specific data is processed but stored in a seperate dataframe.  Long-term this may be used in a supporting model that would feed into the main model based on Date_Time and Address.
    #df_ams output is currently unused.
    """

    df = pd.read_csv(data_file_name)
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])

    #Dataset Builder
    #Criteria
    if farm is not None: df = df[df.Farm == farm]
    if startdate is not None: df = df[df.Date_Time >= startdate]
    if enddate is not None: df = df[df.Date_Time <= enddate]

    df = df[df.Event_Type=='Milking']
    #Routing_Direction
    df = df.rename(columns={'Routing_Direction': 'Seperation'})
    df.Seperation=pd.get_dummies(df.Seperation)['Seperation room']
    #Refusal_Type
    one_hot=pd.get_dummies(df.Refusal_Type)
    df['Refusal_Type_Milk_Visit']=one_hot['Milk Visit']
    df['Refusal_Type_Milk_Interval_Not_Passed']=one_hot['Milk Interval not Passed']
    df['Refusal_Type_Other'] = one_hot.drop(columns=['Milk Visit','Milk Interval not Passed']).sum(axis=1)
    #Visit_Result
    one_hot=pd.get_dummies(df.Visit_Result)
    df['Visit_Result_Successful']=one_hot['Successful']
    df['Visit_Result_Milk_Interval_not_Passed']=one_hot['Milk Interval not Passed']
    df['Visit_Result_Other'] = one_hot.drop(columns=['Successful','Milk Interval not Passed']).sum(axis=1)
    #True/False
    #Current_Lactation
    df['Current_Lactation']=df['Current_Lactation'].astype(int)
    #Failure
    df['Failure']=df['Failure'].astype(int)
    df['Milk_Yield_Dif']=df['Milk_Yield']-df['Milk_Yield_Expected']

    #Index definition
    df = df[df['Animal_Number']!=0]
    df['Index'] = df['Date'].astype(str) + df['Animal_Number'].astype(str)

    df = df.drop(list(df.filter(regex='Unnamed')), axis=1)
    df = df.drop(removal_list,axis=1)

    df = df.sort_values(by=['Index','Date_Time'])

    df.to_csv(os.path.join(output_dir, 'all_legible.csv'))

    df['Date_Time'] = pd.to_numeric(df['Date_Time']) #Converts to ms
    df['Date_Time'] = df['Date_Time'] / 1000 #Reduces size, will scale the same.

    # NA handling
    # na_fill 0=0's,1=mode,2=mean,3=median
    for i in ['0','mode','mean','median']:
        df = na_filler(df, i, var_info[var_info.na_handling==i].var_name.tolist())

    df.to_csv(os.path.join(output_dir, 'all_legible_na_removed.csv'))

    response_list = var_info[var_info.model_category=='response'].var_name.tolist()
    normalize_list = var_info[var_info.scaling=='normalize'].var_name.tolist()

    #This generates the x_data_daily_summary file.  Must be run before normalizer/standardizer.
    daily_summary_bundler(df,output_dir,response_list) #Output [x,12] (9/18/2024).

    # Feature Scaling
    # Handling on Address is inappropriate
    gen_exclusion_list = var_info[var_info.scaling=='exclusion'].var_name.tolist()
    binary_list = var_info[var_info.scaling=='binary'].var_name.tolist()
    standardize_list = var_info[var_info.scaling=='standardize'].var_name.tolist()
    normalize_list = var_info[var_info.scaling == 'normalize'].var_name.tolist()
    #normalize_list = list(set(df.columns) - set(gen_exclusion_list) - set(binary_list) - set(standardize_list))

    df = var_normalizer(df, normalize_list)
    df = var_standardizer(df, standardize_list)

    #Seperate AMS data from milking data
    df_ams = df.drop(columns=milking_list+response_list)
    df = df.drop(columns=ams_list)

    #Write processed data before it becomes completely illegible.
    df.to_csv(os.path.join(output_dir,'all_processed.csv')) #Output [x,31], 1 index, 15 predictors, 15 response variables (9/18/2024)
    df_ams.to_csv(os.path.join(output_dir, 'ams_data.csv')) #Output [x,17] (9/18/2024)

    dataset_bundler_lstm(df,output_dir,response_list)

    return(df,df_ams)

def label_maker(df,response_list,output_dir=''):
    labels = df[['Index'] + response_list].drop_duplicates().sort_values(by=["Index"])
    if output_dir != '': labels.to_csv(os.path.join(output_dir, 'y_data.csv'))
    return(labels)

def dataset_bundler_lstm(df,output_dir,response_list):
    """
    :param df: Dataframe from last step of dataset_preprocessor().
    :param output_dir: Directory to save output to.
    :param response_list: List of response variables.
    :return: Model-ready data (as dataframe), labels (as dataframe.)

    #Takes the pre-processed dataset and groups it into entries for the model.
    #This set-up groups all entries for each Date & animal number combination.  Each row consists of (column_count*entries) columns, with 0 padding equal on the right side.
    """


    #df: pd.read_csv(data_file_name)
    #output_dir: directory x_data/y_data will end up in
    #response_list: list containing response variables

    labels = label_maker(df,response_list,output_dir)
    df = df.drop(columns=response_list).sort_values(by=["Index"])

    #Flatten & Convert to Tensor
    output = list(labels['Index'])
    max_len = 0

    for i in enumerate(labels['Index']):
        df_temp = df[df['Index'] == i[1]]
        df_temp = df_temp.drop(columns=['Index'])
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

def daily_summary_bundler(df,output_dir,response_list):
    """
    :param df: Dataframe from last step of dataset_preprocessor().
    :param output_dir: Directory to save output to.
    :param response_list: List of response variables.
    :return: Model-ready data (as dataframe), labels (as dataframe.)

    #Takes the pre-processed dataset and groups it into entries for the model.
    #This set-up groups all entries for each Date & animal number combination.  In this set-up, there is a pre-set number of variables, 1 per day per animal.
    """

    no_change_var = ["Milkings","Lactation_Days","Lactation_Number",'Total_Eating_Minutes','Rumination_Minutes','Total_Intake_Robot_Feed_1', 'Total_Intake_Robot_Feed_2','Rest_Feed_Robot_Feed_1', 'Rest_Feed_Robot_Feed_2']
    sum_var = ["Milk_Yield","Milk_Yield_Expected"]
    mean_var = ["Last_Milking_Interval"]

    df = df[["Index"] + response_list].groupby("Index").mean().join( #The .mean() prevents dtype issues when joining.
    df[["Index"] + no_change_var].groupby("Index").mean()).join( #Is there a faster function for this?  Just needs first value.
    df[["Index"] + mean_var].groupby("Index").mean()).join(
    df[["Index"]+sum_var].groupby("Index").sum()).reset_index()

    labels = label_maker(df,response_list,output_dir)
    df = df.drop(columns=response_list).sort_values(by=["Index"])
    df = var_normalizer(df, no_change_var+sum_var+mean_var)

    df.drop(columns=["Index"]).to_csv(os.path.join(output_dir, 'x_data_daily_summary.csv'))

    return(df, labels)

def results_logger(dir,fields):
    """
    :param dir: Directory model testing/training is being performed in.
    :param fields: List results are generated from.  Copy below.
    #fields = ['','a'+str(completion_time),x_train.size(0),batchsize,str(model),str(optimizer),str(loss_fn),n_epochs,str(base_learning_rate),str(learning_schedule),float(train_loss.detach()),float(valid_loss.detach()),float(test_loss.detach()),col_names[response_variable_number]]
    :return: None.  Writes to hard-coded location in 'dir'.

    This is an extremely crude logger.  I suggest you switch to something else if able.
    """

    if os.path.isfile(os.path.join(dir, 'Training_Results_Log.csv')):
        with open(os.path.join(dir, 'Training_Results_Log.csv'), 'a', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(fields)
    else:
        pd.DataFrame(columns=['Completion_Time', 'Training_Set_Size', 'Batch_Size', 'Model', 'Optimizer', 'Loss_Function', 'Epochs',
                     'Base_Learning_Rate', 'Schedule', 'Train_Loss', 'Valid_Loss', 'Test_Loss','Response_Variable']).to_csv(os.path.join(dir, 'Training_Results_Log.csv'))
        with open(os.path.join(dir, 'Training_Results_Log.csv'), 'a', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(fields)

def event_tester(df, var_info,farm=None,startdate=None,enddate=None):
    """
    :param df: Dataframe of merged_data_with_feed_filename.
    :return: Tensor ready for model input & labels.

    This is basically a streamlined version of dataset_preprocessor meant for testing a completed model.
    """

    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    if farm is not None: df = df[df.Farm == farm]
    if startdate is not None: df = df[df.Date_Time >= startdate]
    if enddate is not None: df = df[df.Date_Time <= enddate]

    predictor_list = var_info[var_info.model_category=='predictor'].var_name.tolist()
    response_list = var_info[var_info.model_category=='response'].var_name.tolist()
    exclusion_list = var_info[var_info.scaling=='exclusion'].var_name.tolist()
    cat_list = var_info[var_info.scaling=='categorical'].var_name.tolist()
    standardize_list = var_info[var_info.scaling=='standardize'].var_name.tolist()
    normalize_list = var_info[var_info.scaling=='normalize'].var_name.tolist()

    #Filtering
    df = df[df.Event_Type=='Milking']
    df = df[df['Animal_Number']!=0]

    #Add columns
    df['Index'] = df['Date'].astype(str) + df['Animal_Number'].astype(str)
    df['Milk_Yield_Dif']=df['Milk_Yield']-df['Milk_Yield_Expected']

    #Remove excess columns
    df = df[df.columns.intersection(predictor_list+response_list)]
    #Date_Time
    df['Date_Time'] = pd.to_numeric(df['Date_Time']) #Converts to ms
    df['Date_Time'] = df['Date_Time'] / 1000 #Reduces size, will scale the same.

    #Sort.  This is necessary for the bundler to work correctly.
    df = df.sort_values(by=['Index','Date_Time'])

    # NA handling
    for i in range(4):
        df = na_filler(df, i, var_info[var_info.na_handling==i].var_name.tolist())
    df = na_filler(df, 0, df.columns.tolist()) #In case anything was missed

    #Normalize/Standardize
    df = var_normalizer(df, normalize_list)
    df = var_standardizer(df, standardize_list)

    df,labels = event_tester_bundler(df,response_list,10)

    return(df,labels)

def event_tester_bundler(df, response_list,max_events=10):
    """
    :param df: df from event_tester()
    :param max_events: Maximum number of milking events per day to use.
    :return:

    Bundler meant for event_tester function.
    """
    df = df.sort_values(by=["Index"])

    labels = label_maker(df,response_list,'')
    df = df.drop(response_list, axis=1)
    max_len = (df.shape[1]-1)*max_events
    df_output = pd.DataFrame(np.zeros((len(labels),max_len)))

    for i in enumerate(labels.Index.tolist()):
        df_temp = df[df['Index'] == i[1]]
        df_temp = df_temp.drop(columns=['Index'])
        df_temp = df_temp[:max_events]
        df_temp = list(df_temp.to_numpy().flatten())
        df_output.loc[i[0]:i[0], :len(df_temp) - 1] = df_temp

    return (df_output, labels)