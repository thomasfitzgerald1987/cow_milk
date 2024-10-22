from torch import nn

# These were not optimized, only used for viability testing.
# test_Model_linear() was the most heavily used, but if you're working on this,
# I would suggest looking into an LSTM set-up.

class test_Model_lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=15, hidden_size=150, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(150, 300)
        self.linear2 = nn.Linear(300, 150)
        self.linear3 = nn.Linear(150, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        #x = x[:, -1, :]
        x = self.linear3(self.linear2(self.linear1(x)))
        return x

class test_Model_lstm_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=15, hidden_size=150, num_layers=2, batch_first=True)
        self.linear3 = nn.Linear(150, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        #x = x[:, -1, :]
        x = self.linear1(x)
        return x

class test_Model_linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(150, 300)
        self.linear2 = nn.Linear(300, 600)
        self.linear3 = nn.Linear(600, 1200)
        self.linear4 = nn.Linear(1200, 600)
        self.linear5 = nn.Linear(600, 1)
    def forward(self, x):
        x = self.linear5(self.linear4(self.linear3(self.linear2(self.linear1(x)))))
        return x

class test_Model_linear_single(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12, 48)
        self.linear2 = nn.Linear(48, 192)
        self.linear3 = nn.Linear(192, 768)
        self.linear4 = nn.Linear(768, 384)
        self.linear5 = nn.Linear(384, 192)
        self.linear6 = nn.Linear(192, 1)
    def forward(self, x):
        x = self.linear6(self.linear5(self.linear4(self.linear3(self.linear2(self.linear1(x))))))
        return x

class AMS_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=26, hidden_size=260, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(260, 520)
        self.linear2 = nn.Linear(520, 1040)
        self.linear3 = nn.Linear(1040, 520)
        self.linear4 = nn.Linear(520, 260)
        self.linear5 = nn.Linear(260, 4)
    def forward(self, x):
        x, _ = self.lstm(x)
        #x = x[:, -1, :]
        x = self.linear5(self.linear4(self.linear3(self.linear2(self.linear1(x)))))
        return x