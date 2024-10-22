#Libraries
from dataloader import *
from model import *
import torch.utils.data as data
import random

# This file is used for model training.
# It is currently set up for training to predict a single PMR variable at a time (5,6,7,8,9 from y_data.csv)

###EDIT ME###
main_dir = os.getcwd().replace('python_code/','')
output_dir = main_dir + '\\model_testing\\'
output_dir = 'C:\\Users\\Thoma\\OneDrive\\Desktop\\AMS_Data_Project\\Farm_1\\model_testing\\'
###EDIT ME###

random.seed(0)

# Specify float format for pandas tables
pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_columns', 32)

#Get Device


col_names = ['','','','','','Dry_Matter','Lignin', 'Starch', 'CP', 'aNDFom']
pred_names = ['','','','','','Dry_Matter_pred','Lignin_pred', 'Starch_pred', 'CP_pred', 'aNDFom_pred']

for response_variable_number in [5,6,7,8,9]:
    x_train,x_valid,x_test,y_train,y_valid,y_test = load_dataset_from_directory(output_dir, test_prop= .20, valid_prop = .20)
    x_test,y_test = load_full_dataset(output_dir)

    #Define Dataset (for Testing)
    train_size = np.shape(x_train)[0]
    train_size = 5000

    x_train= x_train[0:train_size,:150]
    x_valid= x_valid[:,:150]
    x_test = x_test[:,:150]

    y_train= y_train[0:train_size,response_variable_number:response_variable_number+1] #4:8
    y_valid= y_valid[:,response_variable_number:response_variable_number+1]
    y_test = y_test[:,response_variable_number:response_variable_number+1]

    #For individual line testing
    #epoch = 0 #Debug line
    #x_batch = x_train[0:1,:]
    #y_batch = y_train[0:1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device

    #Instantiate Model and Hyperparameters
    #model = test_Model_lstm().to(device)
    #model = test_Model_lstm_2.to(device)
    model = test_Model_linear().to(device)
    #model = test_Model_linear_single().to(device)
    #model = AMS_Model().to(device)

    #Hyperparameters

    n_epochs = 1000
    base_learning_rate = 0.00001
    learning_rate = base_learning_rate
    learning_schedule = {n_epochs*.1:base_learning_rate/2,
                         n_epochs*.2:base_learning_rate/3,
                         n_epochs*.3:base_learning_rate/5,
                         n_epochs*.5:base_learning_rate/10,
                         n_epochs*.7:base_learning_rate/50,
                         }

    loss_fn = nn.MSELoss() #MSELoss,L1Loss
    batchsize = 25
    optimizer = torch.optim.Adadelta(params=model.parameters(), lr=learning_rate)
    loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=batchsize)
    #print('Model ready.')

    #Training
    x_train, x_valid = x_train.to(device), x_valid.to(device)
    y_train, y_valid = y_train.to(device), y_valid.to(device)
    epoch_count, train_loss_values, valid_loss_values, results = [], [], [], []
    #print("Starting training loop.")
    for epoch in range(n_epochs):
        #print('Beginning epoch ' + str(epoch))
        #print('Running model.train()')
        if epoch in learning_schedule.keys():
            learning_rate = learning_schedule[epoch]
            optimizer = torch.optim.Adadelta(params=model.parameters(), lr=learning_rate)
        model.train()
        for x_batch, y_batch in loader:
        # -1073741819(0xC0000005)
            #if epoch > 0: print('Batch: ',y_batch)
            y_pred = model(x_batch) #May need .squeeze
            #if epoch > 0: print('_pred: ',y_pred)
            loss = loss_fn(y_pred, y_batch)
            #if epoch > 0: print('loss: ',loss)
            #optimizer.zero_grad()
            #if epoch > 0: print('Starting back propagation.')
            #print(psutil.cpu_percent(),psutil.virtual_memory().percent,)
            loss.backward(retain_graph=False)
            #loss.autograd.grad()
            optimizer.step()
        #Validation
        #if epoch % 10 != 0:
        #    continue
        #print('Running model.eval()')
        model.eval()
        #print('model.eval() Complete.')

        with torch.no_grad():
            y_pred = model(x_train) #May need .squeeze()
            train_loss = np.sqrt(loss_fn(y_pred, y_train).detach())
            y_pred = model(x_valid) #May need .squeeze()
            valid_loss = np.sqrt(loss_fn(y_pred, y_valid).detach())
        print("Epoch %d: train RMSE %.4f, validation RMSE %.4f" % (epoch, train_loss, valid_loss))
        epoch_summary = [epoch, float(train_loss), float(valid_loss)]
        results.append(epoch_summary)

    completion_time = datetime.now().strftime("%Y%m%d%H%M")
    output = pd.DataFrame(results, columns=['Epoch', 'Train_Loss', 'Validation_Loss'])
    output.to_csv(os.path.join(output_dir, 'Training_'+completion_time+'_Results.csv'))
    torch.save(model, os.path.join(output_dir, 'Model_'+completion_time+'.pt'))

    y_pred = model(x_test)

    test_actual = pd.DataFrame(y_test.detach().numpy(), columns=[col_names[response_variable_number]])
    test_prediction = pd.DataFrame(y_pred.detach().numpy(), columns=[pred_names[response_variable_number]])
    test_results = pd.concat([test_prediction,test_actual], axis=1)
    test_results.to_csv(os.path.join(output_dir, 'Model_Test_'+completion_time+'_Results.csv'))

    test_loss = np.sqrt(loss_fn(y_pred, y_test).detach())
    print(test_loss)

    fields = ['a'+str(completion_time),x_train.size(0),batchsize,str(model),str(optimizer),str(loss_fn),n_epochs,str(base_learning_rate),str(learning_schedule),float(train_loss.detach()),float(valid_loss.detach()),float(test_loss.detach()),col_names[response_variable_number]]
    results_logger(output_dir,fields)