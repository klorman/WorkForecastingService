import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import transformers as ppb


class CustomDataset(Dataset):
    def __init__(self, table_1, table_2, table_3, padded = None):
        super().__init__()
        # make Bert Model for make embedding requests
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        # Load tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.features_table_1 = ["COL_756", "COL_758", "COL_759", "COL_760", "COL_761", "COL_762", 
                                "COL_763", "COL_764", "COL_769", "COL_770", "COL_771", "COL_772", 
                                "COL_781", "COL_3363"]
        self.features_table_2 = ["Дата закрытия", "Дата создания во внешней системе"]
        self.features_table_3 = ["WORK_NAME", "PLAN_DATE_START", "FACT_DATE_START"] # можно ещё "PLAN_DATE_START", "FACT_DATE_START"
        self.unique_unom = list(set(table_1["COL_782"].apply(int)) & set(table_2["unom"].apply(int)) & set(table_3["UNOM"].apply(int)))
        self.dict_all_works = {name: count for count,name in enumerate(set(table_3["WORK_NAME"]))}
        self.dict_all_works_zero = {name: 0 for name in set(table_3["WORK_NAME"])}
        self.table_1 = table_1
        self.table_2 = table_2
        self.table_3 = table_3
        self.padded = padded #pd.Dself.encoder_text(table_2["Наименование"])
        self.X_batch = []
        self.y_batch = []


    def __len__(self):
        return len(self.unique_unom)


    def __getitem__(self, idx):
        unom = self.unique_unom[idx]
        # idx = unom
        # get needed table

        table_1_unom = self.table_1.loc[(self.table_1["COL_782"].apply(int)==unom)]
        table_2_unom = self.table_2.loc[(self.table_2["unom"].apply(int)==unom)]
        table_3_unom = self.table_3.loc[(self.table_3["UNOM"].apply(int)==unom)]

        work_unom_time = self.get_difference_time_work(table_3_unom)

        table_3_unom['WORK_NAME'] = table_3_unom['WORK_NAME'].apply((lambda x: self.dict_all_works[x]))
        # choose needed coloumn in table
        table_1_res = self.get_table(table_1_unom, self.features_table_1).apply(float).to_numpy()
        table_2_res = self.get_table(table_2_unom, self.features_table_2)
        table_2_res = self.convert_date(table_2_res)
        #print("table_2_res", table_2_res[0][0])
        table_3_res = self.get_table(table_3_unom, self.features_table_3)
        #print("table_3_res", table_3_res)
        table_2_text = self.get_padded_sent(unom)
        #print("table_2_text", table_2_text)
        # get right result
        print(work_unom_time)
        #work_unom_time = self.get_difference_time_work(table_3_unom)
        #print(table_1_res, table_2_res, table_3_res)
        self.X_batch.append((table_1_res, table_2_res, table_3_res, table_2_text))
        self.y_batch.append(pd.DataFrame(work_unom_time).to_numpy())
        return  0,0#torch.tensor((table_1_res, table_2_res, table_3_res, table_2_text)), torch.tensor(work_unom_time)
    

    def convert_date(self, table):
        new_t = {"day_beg": [], "month_beg": [], "year_beg": [], "day_end": [], "month_end": [], "year_end": []}
        new_t["day_beg"] = table["Дата создания во внешней системе"].dt.day.to_numpy()
        new_t["month_beg"] = table["Дата создания во внешней системе"].dt.month.to_numpy()
        new_t["year_beg"]  = table["Дата создания во внешней системе"].dt.year.to_numpy()
        new_t["hour_beg"]  = table["Дата создания во внешней системе"].dt.hour.to_numpy()
        new_t["minute_beg"]  = table["Дата создания во внешней системе"].dt.minute.to_numpy()
        new_t["day_end"] = table["Дата закрытия"].dt.day.to_numpy()
        new_t["month_end"] = table["Дата закрытия"].dt.month.to_numpy()
        new_t["year_end"] = table["Дата закрытия"].dt.year.to_numpy()
        new_t["hour_end"] = table["Дата закрытия"].dt.hour.to_numpy()
        new_t["minute_end"] = table["Дата закрытия"].dt.minute.to_numpy()
        return pd.DataFrame(new_t)
    

    def get_padded_sent(self, unom: int):
        index = self.table_2[self.table_2['unom'].apply(int) == unom].index 
        pad_sent = []
        for idx in index:
            pad_sent.append(self.padded[idx])
        return np.asarray(pad_sent)


    def get_table(self, table, features):
        dict_new_table = {name: [] for name in features}
        for name in features:
            dict_new_table[name] = table[name]
        return pd.DataFrame(dict_new_table)
    

    def get_difference_time_work(self, table): # передаем данные по конкретному уному
        # get list all work
        work_unom_time = self.dict_all_works_zero
        # get date fact and plan work
        fact_beg_day = table["FACT_DATE_START"].dt.day
        fact_beg_month = table["FACT_DATE_START"].dt.month
        fact_beg_year  = table["FACT_DATE_START"].dt.year
        plan_beg_day = table["PLAN_DATE_START"].dt.day
        plan_beg_month = table["PLAN_DATE_START"].dt.month
        plan_beg_year = table["PLAN_DATE_START"].dt.year
        # compute result difference 
        result = ((plan_beg_year - fact_beg_year) * 365 + (plan_beg_month - fact_beg_month) * 30 + (plan_beg_day - fact_beg_day)).to_numpy()
        # compute mean time deviation for all works
        for count, name in enumerate(table["WORK_NAME"]):
            work_unom_time[name] =  (work_unom_time[name] + result[count]) / 2
        values = []
        for key in work_unom_time.keys():
            values.append(work_unom_time[key])
        return values
    

    def encoder_text(self, coloumn_text):
        # apply tokenizer
        with torch.no_grad():
            coloumn = coloumn_text.apply((lambda x: self.tokenizer.encode(x, add_special_tokens=True)))
        # <pad>
        max_len = 0
        for i in coloumn.values:
            if len(i) > max_len:
                max_len = len(i)

        if max_len % 2 == 1:
            max_len += 1

        padded = np.array([i + [0]*(max_len-len(i)) for i in coloumn.values])

        return pd.DataFrame(padded)


# Каждая заявка вносит свой вклад в приближение капитального ремонта
class ProcessRequestsNN(nn.Module):
    def __init__(self, num_features: int, num_add_features: int, hidden_dim: int = 64, emb_dim: int = 1024, modelBERT = None, num_requests: int = 19,
                 dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embedding_size = emb_dim
        self.num_requests = num_requests
        self.output_conv = self.__compute_output() + num_add_features
        # Layers
        # Embedding
        if modelBERT is None:
            # make Bert Model for make embedding requests
            model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
            self.embed = model_class.from_pretrained(pretrained_weights)
            self.dimBERT = 768
        else:
            self.embed = modelBERT
            self.dimBERT = 768
        
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=4)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.max_pool_3 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.max_pool_4 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        
        self.flatten = nn.Flatten()


        self.embed_after_conv = nn.Embedding(self.output_conv , self.embedding_size)
        self.linear_1 = nn.Linear(self.output_conv, int(self.embedding_size/2))
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(int(self.embedding_size/2), int(self.embedding_size/8))
        self.linear_3 = nn.Linear(int(self.embedding_size/8), self.num_requests)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(self.output_conv)

    
    def forward(self, in_data, in_data_2):
        attention_mask = torch.tensor(np.where(in_data != 0, 1, 0))
        #embed = model_class.from_pretrained(pretrained_weights)
        with torch.no_grad():
            out = self.embed(in_data, attention_mask=attention_mask)

        out = out[0].reshape(out[0].shape[0], out[0].shape[1]*out[0].shape[2]).unsqueeze(1)
        print("out.shape", out.shape)

        # Apply conv layer
        out = self.conv_1(out)
        out = self.max_pool_1(out)
        out = self.conv_2(out)
        out = self.max_pool_2(out)
        out = self.conv_3(out)
        out = self.max_pool_3(out)
        out = self.conv_4(out)
        out = self.max_pool_4(out)
        out = self.flatten(out)

        # concatenate date and name requests
        out = torch.concatenate(((out, in_data_2.permute(1,0))), dim=1)
        out = out.unsqueeze(0)
        # Apply Fully connected layer
        #out = self.batch_norm(out)
        #out = self.embed_after_conv(out.Lomnn)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear_3(out)

        return out
    

    def __compute_output(self):
        dim1 = 26882 
        dim_out = dim1
        return dim_out


class ComputeResultNN(nn.Module):
    def __init__(self, num_features_req: int, num_features_cat: int = 14, num_output: int = 19,
                 emb_size: int = 128, hidden_dim: int = 64, dropout: float = 0.1, num_add_features: int = 10,
                 num_features_date_cup:int = 4):
        super().__init__()
        self.num_features_cat = num_features_cat
        self.emb_size = emb_size
        self.hid_dim = hidden_dim

        self.process_request = ProcessRequestsNN(num_features=num_features_req,
                                                 num_add_features=num_add_features)
        
        self.rnn = nn.LSTM(input_size = num_output,
                           hidden_size = self.hid_dim,
                           num_layers = 1, # may be 4 or 5 поставить
                           dropout=dropout,
                           bidirectional=True) #через lstm прогоняем все запросы получаем скрытое представление
        self.linear_1 = nn.Linear(num_features_cat + num_output, self.hid_dim)
        self.linear_2 = nn.Linear(self.hid_dim, 1)
        self.relu = nn.ReLU()
    

    def forward(self, in_data_cat, in_data_request, in_data_date):
        # proccess request
        out_req = self.process_request(in_data_request, in_data_date).squeeze(0)
        #out_req, (hid, cell) = self.rnn(out_req)
        # concatenate information
        multiply = torch.tensor([1 for i in range(19)])
        for i in out_req:
            multiply = torch.multiply(multiply, i)
        print(multiply.size())
        data_concat = torch.concatenate((multiply, in_data_cat), dim=0)

        # proccess specifications and out other model
        out = self.linear_1(data_concat.float())
        out = self.relu(out)
        out = self.linear_2(out)

        return out


MAX_LEN = 70
NUM_FEATURES_REQ = MAX_LEN


# получение даты и ее разности для запросов
def get_data_date(table):
    days_close = table["day_end"]

    time_perform_days = table["day_end"].to_numpy() - table["day_beg"]
    time_perform_month = table["month_end"].to_numpy() - table["month_beg"].to_numpy()
    time_perform_year = table["year_end"].to_numpy() - table["year_beg"].to_numpy()
    time_perform_hour = table["hour_end"].to_numpy() - table["hour_beg"].to_numpy()
    time_perform_minute = table["minute_end"].to_numpy() - table["minute_beg"].to_numpy()

    arr_times = [table["day_beg"].to_numpy(), table["month_beg"].to_numpy(), table["year_beg"].to_numpy(), 
                 table["hour_beg"].to_numpy(), table["minute_beg"].to_numpy(),time_perform_days,
                 time_perform_month, time_perform_year,
                 time_perform_hour,time_perform_minute]
    return torch.tensor(arr_times)


# вычисление ответа
def compute_y(table):
    d = {f'{i}': 0 for i in range(19)}
    d_exists = {f'{i}': 0 for i in range(19)}
    d_out = {}
    fact_beg_day = table["FACT_DATE_START"].dt.day.to_numpy()
    fact_beg_month = table["FACT_DATE_START"].dt.month.to_numpy()
    fact_beg_year  = table["FACT_DATE_START"].dt.year.to_numpy()
    plan_beg_day = table["PLAN_DATE_START"].dt.day.to_numpy()
    plan_beg_month = table["PLAN_DATE_START"].dt.month.to_numpy()
    plan_beg_year = table["PLAN_DATE_START"].dt.year.to_numpy()
    result = ((plan_beg_year - fact_beg_year) * 365 + (plan_beg_month - fact_beg_month) * 30 + (plan_beg_day - fact_beg_day))
    
    table = table.to_numpy()
    for count, name in enumerate(table):
        d[f"{name[0]}"] = (d[f"{name[0]}"] + result[count]) / 2
        d_exists[f"{name[0]}"] = 1
    for k in d.keys():
        if d_exists[k] == 1:
            d_out[k] = d[k]
    return d_out


checkpoint_path = "predict_dev.pth"


def save_checkpoint(checkpoint_path, model, optimizer):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizer’s states and hyperparameters used.
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def get_model_and_opt():
    model = ComputeResultNN(num_features_req=NUM_FEATURES_REQ,
                            num_features_cat=14,
                            num_output=19,
                            num_add_features=10,
                            num_features_date_cup=7)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    load_checkpoint(checkpoint_path, model, opt)
    return model, opt


def get_padded(table_2, maxl_len=70):
    #BERT tokenizer
    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    # удаление нахер вссего NaN дерьма
    for count, elem in enumerate(table_2["Наименование"].isna()):
        if elem:
            table_2 = table_2.drop(labels=[count], axis=0)
    table_2["Наименование"].isna().sum()

    # apply tokenizator
    df = table_2["Наименование"].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    # find max_len
    max_len = 0
    for i in df.values:
        if len(i) > max_len:
            max_len = len(i)

    if max_len % 2 == 1:
        max_len += 1

    # add <pad> symbol
    padded = np.array([i + [0]*(max_len-len(i)) for i in df.values])
    return padded


# in_data_cat - категориальные признаки + дата окончания последнего кап ремонта (14 штук)
# in_data_requests - заявки по данному уному (заявки по уному в виде таблицы)
# in_data_date - данные о времени заявок (3 т)
def get_result(table_1, table_2, table_3, model):
    model.train(False)
    result = {}
    padded = get_padded(table_2)
    dataset = CustomDataset(table_1, table_2, table_3, padded)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    for x,_ in dataloader:
        for count, x_batch in enumerate(dataset.X_batch):
            y = compute_y(x_batch[2])
            for work_id in y.keys():
                data_request = x_batch[3] #данные по заявкам
                data_add_for_first_model = get_data_date(x_batch[1]) # данные времени по заявкам
                data_mlh = x_batch[0] # хар-ки дома
                number_work = float(work_id) # номер работы
                y_true = y[work_id]

                #in_data_cat - категориальные признаки + дата окончания последнего кап ремонта
                #in_data_cat = torch.cat((torch.tensor(data_mlh), torch.tensor([int(work_id)])))
                in_data_cat = torch.tensor(data_mlh)
                in_data_cat[13] = int(work_id)
                # in_data_requests - заявки по данному уному
                in_data_requests = torch.tensor(data_request)
                # in_data_date - данные о времени заявок
                in_data_date = torch.tensor(data_add_for_first_model)
                
                preds = model(in_data_cat, in_data_requests, in_data_date)

                result[work_id] = preds
        dataset.X_batch = []
        dataset.y_batch = []
    return result


def additional_training(table_1, table_2, table_3, opt, model):
    model.train(True)
    loss_fn = nn.L1Loss()
    padded = get_padded(table_2)
    dataset = CustomDataset(table_1, table_2, table_3, padded)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    for x,_ in dataloader:
        for count, x_batch in enumerate(dataset.X_batch):
            y = compute_y(x_batch[2])
            print(y)
            for work_id in y.keys():
                data_request = x_batch[3] #данные по заявкам
                data_add_for_first_model = get_data_date(x_batch[1]) # данные времени по заявкам
                data_mlh = x_batch[0] # хар-ки дома
                number_work = float(work_id) # номер работы
                y_true = y[work_id]

                #in_data_cat - категориальные признаки + дата окончания последнего кап ремонта
                in_data_cat = torch.cat((torch.tensor(data_mlh), torch.tensor([int(work_id)])))

                # in_data_requests - заявки по данному уному
                in_data_requests = torch.tensor(data_request)
                # in_data_date - данные о времени заявок
                in_data_date = torch.tensor(data_add_for_first_model)

                preds = model(in_data_cat, in_data_requests, in_data_date)

                opt.zero_grad()
                loss = loss_fn(torch.tensor(y_true), preds)
                loss.backward()
                opt.step()
    model.train(False)
    return model, opt
