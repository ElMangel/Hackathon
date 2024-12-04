import sys
import pandas as pd
import numpy as np

def load_dataset(active):
    import yfinance as yf
    import pandas as pd
    dataset = yf.download(active, start='2019-01-01', end='2024-12-02')
    cols = ['Close']
    dataset = dataset[cols]
    dataset.columns = [x[1] for x in list(dataset.columns)]
    dataset = dataset.asfreq('D', method='ffill')

    return dataset

def forecast_chronos(df, prediction_length, ci=0.9, complexity='small'):
    '''
    data: numpy 1d array
    complexity: ['tiny','mini','small','base','large']
    '''
    import pandas as pd
    import numpy as np
    import torch
    from chronos import ChronosPipeline

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    e = 1-ci
    data = df.to_numpy().flatten()  
    data_tensor = torch.tensor(data)
    pipeline = ChronosPipeline.from_pretrained(
                'amazon/chronos-t5-'+complexity,
                device_map=device,
                torch_dtype=torch.bfloat16,
                )
    forecast = pipeline.predict(data_tensor, prediction_length)[0].numpy()
    low, median, high = np.quantile(forecast, [e/2, 0.5, 1-e/2], axis=0)
    return low, median, high

def forecast_granite(df, prediction_length, ci=0.9, dataset_name='model', batch_size=64, seed=42):
    import math
    import os
    import tempfile

    import pandas as pd
    import numpy as np
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
    from transformers.integrations import INTEGRATION_TO_CALLBACK

    from tsfm_public import TimeSeriesPreprocessor, TrackingCallback, count_parameters, get_datasets
    from tsfm_public.toolkit.get_model import get_model
    from tsfm_public.toolkit.lr_finder import optimal_lr_finder
    df = df.rename(columns={df.columns[0]: 'target'})
    #df = pd.DataFrame(dataset, columns=['target'])
    e = 1-ci

    # TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
    TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r1"
    # TTM_MODEL_PATH = "ibm/ttm-research-r2"

    CONTEXT_LENGTH = 512
    OUT_DIR = "ttm_finetuned_models/"
    id_columns = []
    target_columns = ['target']

    column_specifiers = {
      #"timestamp_column": timestamp_column,
      "id_columns": id_columns,
      "target_columns": target_columns,
      "control_columns": [],
    }

    split_config = {
      "train": 0.8,
      "test": 0.1
      }

    tsp = TimeSeriesPreprocessor(
      **column_specifiers,
      context_length=CONTEXT_LENGTH,
      prediction_length=prediction_length,
      scaling=False,
      encode_categorical=False,
      scaler_type="standard",
    )
    dset_train, dset_valid, dset_test = get_datasets(tsp, df, split_config)

    # Load model
    zeroshot_model = get_model(TTM_MODEL_PATH, context_length=CONTEXT_LENGTH, prediction_length=prediction_length)

    temp_dir = tempfile.mkdtemp()
    # zeroshot_trainer
    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=batch_size,
            seed=seed,
            report_to="none",
        ),
    )
    # evaluate = zero-shot performance
    #print("+" * 20, "Test MSE zero-shot", "+" * 20)
    zeroshot_output = zeroshot_trainer.evaluate(dset_test)
    #print(zeroshot_output)

    # get predictions
    predictions_dict = zeroshot_trainer.predict(dset_test)
    predictions_np = predictions_dict.predictions[0]

    #print(predictions_np.shape)
    low, median, high = np.quantile(predictions_np[:,:,0], [e/2, 0.5, 1-e/2], axis=0)

    return low, median, high

def forecast_timegpt(df, prediction_length, ci=0.9):
    from nixtla import NixtlaClient
    dftgpt = df.rename(columns={df.columns[0]: 'y'}).reset_index().rename({'Date':'ds'}, axis=1)
    train_data = dftgpt[:-prediction_length]
    test_data = dftgpt[-prediction_length:]

    nixtla_client = NixtlaClient(api_key ='nixak-r6loGX4nnn4klY1y9tYi8zrlPeKyfcj54tQIWruze3EDrIwqjvuTediQMNubSZg9T7vj3Bg5OKJSp4Pt')
    # Forecasting
    fcst_df = nixtla_client.forecast(train_data, h=prediction_length, level=[ci*100])
    fcst = fcst_df.iloc[:,1:].to_numpy()
    low, median, high = fcst[:,2], fcst[:,0], fcst[:,1]
    return low, median, high

def forecast_hybrid(df, prediction_length, fcst_chronos, fcst_granite, fcst_gpt, rmse_chronos, rmse_granite, rmse_gpt):
    inv_rmse_array = np.array([1/rmse_chronos, 1/rmse_granite, 1/rmse_gpt])
    weights = (inv_rmse_array/inv_rmse_array.sum()).reshape(-1,1)

    # median
    fcst_ch_m = fcst_chronos[1].reshape(-1,1)
    fcst_gr_m = fcst_granite[1].reshape(-1,1)
    fcst_gpt_m = fcst_gpt[1].reshape(-1,1)
    fcst = np.hstack([fcst_ch_m, fcst_gr_m, fcst_gpt_m])
    fcst = (fcst@weights).sum(axis=1)

    # low
    fcst_ch_l = fcst_chronos[0].reshape(-1,1)
    fcst_gr_l = fcst_granite[0].reshape(-1,1)
    fcst_gpt_l = fcst_gpt[0].reshape(-1,1)
    fcst_l = np.hstack([fcst_ch_l, fcst_gr_l, fcst_gpt_l])
    fcst_l = (fcst_l@weights).sum(axis=1)

    # high
    fcst_ch_h = fcst_chronos[2].reshape(-1,1)
    fcst_gr_h = fcst_granite[2].reshape(-1,1)
    fcst_gpt_h = fcst_gpt[2].reshape(-1,1)
    fcst_h = np.hstack([fcst_ch_h, fcst_gr_h, fcst_gpt_h])
    fcst_h = (fcst_h@weights).sum(axis=1)  

    test_data = df.to_numpy().flatten()[-prediction_length:]
    model_rmse = rmse(test_data, fcst)
    print(weights)
    return fcst_l, fcst, fcst_h

def rmse(true, pred):
    import numpy
    return np.sqrt(np.sum(((true.flatten()-pred.flatten())**2))/len(true))

def evaluate_model(model, df, prediction_length=12, ci=0.9, complexity='tiny'):
    train_data = df[:-prediction_length]
    test_data = df[-prediction_length:]

    if model=='chronos':
      forecast = forecast_chronos(train_data, prediction_length, ci, complexity=complexity)
    elif model=='granite':
      forecast = forecast_granite(train_data, prediction_length, ci, 'granite')
    elif model=='timegpt':
      forecast = forecast_timegpt(train_data, prediction_length, ci)
    model_rmse = rmse(test_data.to_numpy().flatten(), forecast[1])
    return forecast, model_rmse

def forecast_final(model, df, rmse_ch, rmse_gr, rmse_gpt, prediction_length=12, ci=0.9, complexity='tiny'):
    train_data = df

    if model=='chronos':
        forecast = forecast_chronos(train_data, prediction_length, ci, complexity=complexity)
    elif model=='granite':
        forecast = forecast_granite(train_data, prediction_length, ci, 'granite')
    elif model=='timegpt':
        forecast = forecast_timegpt(train_data, prediction_length, ci)
    elif model=='hybrid':
        fcst_ch = forecast_chronos(train_data, prediction_length, ci, complexity=complexity)
        fcst_gr = forecast_granite(train_data, prediction_length, ci, 'granite')
        fcst_gpt = forecast_timegpt(train_data, prediction_length, ci)
        forecast = forecast_hybrid(train_data, prediction_length, fcst_ch, fcst_gr, fcst_gpt, rmse_ch, rmse_gr, rmse_gpt)
    return forecast

def prediction(df, prediction_length=12, ci=0.9, complexity='tiny'):
    test_data = df[-prediction_length:]
    fcst_ch, rmse_ch = evaluate_model('chronos', df, prediction_length, ci, complexity=complexity)
    fcst_gr, rmse_gr = evaluate_model('granite', df, prediction_length, ci)
    fcst_gpt, rmse_gpt = evaluate_model('timegpt', df, prediction_length, ci)
    fcst_hy = forecast_hybrid(df, prediction_length, fcst_ch, fcst_gr, fcst_gpt, rmse_ch, rmse_gr, rmse_gpt)
    rmse_hy = rmse(test_data.to_numpy().flatten(), fcst_hy[1])
    rmse_array = np.array([rmse_ch, rmse_gr, rmse_gpt, rmse_hy])
    idx_winner = np.argmin(rmse_array)
    fcst_order = {0:fcst_ch, 1:fcst_gr, 2:fcst_gpt, 3:fcst_hy}
    name_order = {0:'chronos',1:'granite',2:'timegpt',3:'hybrid'}

    winner = name_order[idx_winner]
    print('Winner: ',winner)
    print('RMSE: ',np.min(rmse_array))
    
    forecast = forecast_final(winner, df, rmse_ch, rmse_gr, rmse_gpt, prediction_length, ci, complexity)
    print('Forecast: ',forecast) 
    #print('Forecast: ',fcst_order[idx_winner])

    return name_order[idx_winner], forecast, np.min(rmse_array)

def plot_predict(dataset, forecast, forecast_length, ci):
    import matplotlib.pyplot as plt
    import numpy as np
    low, median, high = forecast[0], forecast[1], forecast[2]

    label_plot = str(int(ci*100))+'% confidence interval'
    forecast_index = range(len(dataset)-forecast_length, len(dataset)-forecast_length+len(median))
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(dataset)), dataset, color="royalblue", label="historical data")
    plt.plot(forecast_index, median, color="tomato", label="median forecast")
    plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label=label_plot)
    plt.legend(loc=0)
    plt.grid()
    plt.show

if __name__ == '__main__':

    if len(sys.argv) !=5:
        print('Usage: python predict.py <action_name> <steps> <confidence interval> <complexity>')
        sys.exit(1)
    
    action_name = sys.argv[1]
    steps = int(sys.argv[2])
    ci = float(sys.argv[3])
    complexity = sys.argv[4]

    df = load_dataset(action_name)
    prediction(df, steps, ci, complexity)



