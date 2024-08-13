import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import pandas as pd
from core import standard_weighted_quantile, trailing_window, aci, aci_clipped, quantile, quantile_integrator_log, quantile_integrator_log_scorecaster
from core.synthetic_scores import generate_scores
from core.model_scores import generate_forecasts
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datasets import load_dataset
from darts import TimeSeries
import yaml
import pickle
import pdb
import matplotlib.pyplot as plt


def compute_rmse(y, forecasts):
    return np.sqrt(mean_squared_error(y, forecasts))

def compute_mae(y, forecasts):
    return mean_absolute_error(y, forecasts)

def compute_r2(y, forecasts):
    return r2_score(y,forecasts)
    
def compute_mape(y, forecasts):
    return np.mean(np.abs((y - forecasts) / y)) * 100

def compute_wis(y, forecasts):
    in_interval = np.logical_and(y >= forecasts[0], y <= forecasts[1])
    interval_width = forecasts[1] - forecasts[0]
    return np.mean(interval_width * np.logical_not(in_interval))
    
#def compute_prediction_intervals(y, forecasts):
    #lower_bound = np.minimum(y, forecasts)
    #upper_bound = np.maximum(y, forecasts)
    #return lower_bound, upper_bound
    
def compute_prediction_intervals(y, forecasts, confidence_level=0.95):
    residuals = y - forecasts
    std_dev = np.std(residuals)
    critical_value = 1.96
    margin_of_error = critical_value * std_dev
    
    lower_bound = forecasts - margin_of_error
    upper_bound = forecasts + margin_of_error
    
    return lower_bound, upper_bound, margin_of_error
    
def visualize_prediction_intervals(data, lower_bound, upper_bound, save_path='prediction_intervals.pdf'):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['y'], label='True Values', color='blue')
    plt.plot(data.index, data['forecasts'], label='Forecasted Values', color='green')
    plt.fill_between(data.index, lower_bound, upper_bound, color='gray', alpha=0.5, label='Prediction Interval')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title('Forecasted Values with Prediction Intervals')
    plt.legend()
    plt.savefig(save_path)  # Save the figure as a PDF
    plt.show()
    
if __name__ == "__main__":
    json_name = sys.argv[1]
    if len(sys.argv) > 2:
        overwrite = sys.argv[2].split(",")
    else:
        overwrite = []
    args = yaml.safe_load(open(json_name))
    # Set up folder and filename
    foldername = './results/'
    config_name = json_name.split('.')[-2].split('/')[-1]
    filename = foldername + config_name + ".pkl"
    os.makedirs(foldername, exist_ok=True)
    real_data = args['real']
    quantiles_given = args['quantiles_given']
    multiple_series = args['multiple_series']
    score_function_name = args['score_function_name'] if real_data else "synthetic"
    model_names = args['sequences'][0]['model_names'] if real_data else ["synthetic"]
    ahead = args['ahead'] if real_data and 'ahead' in args.keys() else 1
    minsize = args['minsize'] if real_data and 'minsize' in args.keys() else 0
    asymmetric = False

    # Try reading in results
    try:
        with open(filename, 'rb') as handle:
            all_results = pickle.load(handle)
    except:
        all_results = {}

    for model_name in model_names:
        try:
            results = all_results[model_name]
        except:
            results = {}
        
        # Initialize lists to store average scores
        avg_rmse_scores = []
        avg_mae_scores = []
        avg_r2_scores = []
        avg_mape_scores = []
        avg_wis_scores = []

        for model_name in model_names:
            try:
                results = all_results[model_name]
            except:
                results = {}
        # Initialize the score function
        if real_data:
            score_function_name = args['score_function_name']
            if score_function_name == "absolute-residual":
                def score_function(y, forecast):
                    return np.abs(y - forecast)
                def set_function(forecast, q):
                    return np.array([forecast - q, forecast + q])
            elif score_function_name == "signed-residual":
                def score_function(y, forecast):
                    return np.array([forecast - y, y - forecast])
                def set_function(forecast, q):
                    return np.array([forecast - q[0], forecast + q[1]])
                asymmetric = True
            elif score_function_name == "cqr-symmetric":
                def score_function(y, forecasts):
                    return np.maximum(forecasts[0] - y, y - forecasts[-1])
                def set_function(forecast, q):
                    return np.array([forecast[0] - q, forecast[-1] + q])
            elif score_function_name == "cqr-asymmetric":
                def score_function(y, forecasts):
                    return np.array([forecasts[0] - y, y - forecasts[-1]])
                def set_function(forecast, q):
                    return np.array([forecast[0] - q[0], forecast[-1] + q[1]])
                asymmetric = True
            else:
                raise ValueError("Invalid score function name")

        # Get dataframe and add forecasts and scores to it
        if real_data:
            data = load_dataset(args['sequences'][0]['dataset'])
            # Get the forecasts
            if 'forecasts' not in data.columns:
                os.makedirs('./datasets/proc/', exist_ok=True)
                os.makedirs('./datasets/proc/' + config_name, exist_ok=True)
                args['sequences'][0]['savename'] = './datasets/proc/' + config_name +  '/' + model_name + '.npz'
                args['sequences'][0]['T_burnin'] = args['T_burnin']
                args['sequences'][0]['ahead'] = ahead
                args['sequences'][0]['model_name'] = model_name
                data['forecasts'] = generate_forecasts(data, **args['sequences'][0])
            # Compute scores
            if 'scores' not in data.columns:
                data['scores'] = [ score_function(y, forecast) for y, forecast in zip(data['y'], data['forecasts']) ]
        else:
            scores_list = []
            for key in args['sequences'].keys():
                scores_list += [generate_scores(**args['sequences'][key])]
            scores = np.concatenate(scores_list).astype(float)
            # Make a pandas dataframe with a datetime index and the scores in their own column called `scores'.
            data = pd.DataFrame({'scores': scores}, index=pd.date_range(start='1/1/2018', periods=len(scores), freq='D'))

        # Loop through each method and learning rate, and compute the results
        for method in args['methods'].keys():
            if (method in results.keys()) and (method not in overwrite):
                continue
            fn = None
            if method == "Trail":
                fn = trailing_window
                args['methods'][method]['lrs'] = [None]
            elif method == "ACI":
                fn = aci
            elif method == "ACI (clipped)":
                fn = aci_clipped
            elif method == "Quantile":
                fn = quantile
            elif method == "Quantile+Integrator (log)":
                fn = quantile_integrator_log
            elif method == "Quantile+Integrator (log)+Scorecaster":
                fn = quantile_integrator_log_scorecaster
            else:
                raise Exception(f"Method {method} not implemented")
            lrs = args['methods'][method]['lrs']
            kwargs = args['methods'][method]
            kwargs["T_burnin"] = args["T_burnin"]
            kwargs["data"] = data if real_data else None
            kwargs["seasonal_period"] = args["seasonal_period"] if "seasonal_period" in args.keys() else None
            kwargs["config_name"] = config_name
            kwargs["ahead"] = ahead
            # Compute the results
            results[method] = {}
            for lr in lrs:
                if asymmetric:
                    stacked_scores = np.stack(data['scores'].to_list())
                    kwargs['upper'] = False
                    q0 = fn(stacked_scores[:,0], args['alpha']/2, lr, **kwargs)['q']
                    kwargs['upper'] = True
                    q1 = fn(stacked_scores[:,1], args['alpha']/2, lr, **kwargs)['q']
                    q = [ np.array([q0[i], q1[i]]) for i in range(len(q0)) ]
                else:
                    kwargs['upper'] = True
                    q = fn(data['scores'].to_numpy(), args['alpha'], lr, **kwargs)['q']
                if real_data:
                    sets = [ set_function(data['forecasts'].interpolate().to_numpy()[i], q[i]) for i in range(len(q)) ]
                    # Make sure the set size is at least minsize by setting sets[j][0] = min(sets[j][0], sets[j][1]-minsize) and sets[j][1] = max(sets[j][1], sets[j][1]+minsize)
                    sets = [ np.array([np.minimum(sets[j][0], sets[j][1]-minsize), np.maximum(sets[j][1], sets[j][0]+minsize)]) for j in range(len(sets)) ]
                else:
                    sets = None
                results[method][lr] = { "q": q, "sets": sets }

        # Save some metadata
        results["scores"] = data['scores']
        results["alpha"] = args['alpha']
        results["T_burnin"] = args['T_burnin']
        results["quantiles_given"] = quantiles_given
        results["multiple_series"] = multiple_series
        results["real_data"] = real_data
        results["score_function_name"] = score_function_name
        results["asymmetric"] = asymmetric

        if real_data:
            results["forecasts"] = data['forecasts']
            results["data"] = data
        all_results[model_name] = results

    # Save results
    with open(filename, 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Initialize an empty list to store the widths of prediction intervals
    prediction_interval_widths = []
    
    # Initialize lists to store lower and upper bounds for averaging
    all_lower_bounds = []
    all_upper_bounds = []
        
    # Iterate over each model name
    for model_name in model_names:
        results = all_results[model_name]
        data = results["data"]
    
        # Compute the metrics
        rmse = compute_rmse(data['y'], data['forecasts'])
        mae = compute_mae(data['y'], data['forecasts'])
        mape = compute_mape(data['y'], data['forecasts'])
        r2 = compute_r2(data['y'], data['forecasts'])
        
        forecasts_values = data['forecasts'].values
        if forecasts_values.ndim == 1:
            forecasts_values = forecasts_values.reshape(-1, 1)
        #wis = compute_wis(data['y'], (np.min(forecasts_values, axis=1), np.max(forecasts_values, axis=1)))
        
         # Compute the lower and upper bounds of the prediction intervals
        #lower_bound, upper_bound = compute_prediction_intervals(data['y'],data['forecasts'])
        lower_bound, upper_bound, margin_of_error = compute_prediction_intervals(data['y'], data['forecasts'])
        visualize_prediction_intervals(data, lower_bound, upper_bound)
         # Print the lower and upper bounds along with their corresponding forecasts
        for i, (forecast, lower, upper) in enumerate(zip(forecasts_values, lower_bound, upper_bound)):
            print(f"Forecast {i+1}: {forecast}, Lower Bound: {lower}, Upper Bound: {upper}")
        
        # Compute the width of each prediction interval
        interval_widths = upper_bound - lower_bound
    
        # Append the widths to the list
        prediction_interval_widths.extend(interval_widths)
        
        # Append the lower and upper bounds to their respective lists
        all_lower_bounds.extend(lower_bound)
        all_upper_bounds.extend(upper_bound)


        # Calculate the average width of prediction intervals
        avg_interval_width = np.mean(prediction_interval_widths)
        
        # Print the evaluation results for the current model
        print("Evaluation Results for", model_name)
        print("Average RMSE:", np.mean(rmse))
        print("Average MAE:", np.mean(mae))
        print("Average MAPE:", np.mean(mape))
        print("Average R2:", np.mean(r2))
        #print("Average WIS:", np.mean(wis))
        print("Average Width of Prediction Intervals:", avg_interval_width)
        # Print the overall lower and upper bounds
        average_lower_bound = np.mean(all_lower_bounds)
        average_upper_bound = np.mean(all_upper_bounds)

        print("Average Lower Bound:", average_lower_bound)
        print("Average Upper Bound:", average_upper_bound)
        print("\n")