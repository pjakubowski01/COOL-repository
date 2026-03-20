#__all__ = ["loadDataFluvius", "train_all", "loadTrainedModels", "method_a_optimization", "method_b_optimization", "method_c_optimization", "calculate_methodINM", "calculate_methodINM_POA", "splitData", "predict_and_plot", "visualizeSubstationLoad"]

from tabnanny import verbose

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
import time

import xgboost as xgb

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from random import choice, uniform
from copy import deepcopy


from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.svm import SVR

from scipy.stats import pearsonr

from pvlib import pvsystem, irradiance
from pvlib.location import Location
from pvlib.modelchain import ModelChain

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import cvxpy as cp

from tqdm import tqdm, trange

warnings.simplefilter("ignore", category=ConvergenceWarning)

Power = 4     # Average Belgium Rooftop PV Capacity

gamma = -0.003

azimuths = np.linspace(90, 270, 7)      # From East to West
tilts = np.linspace(20, 80, 3)          # From low to high tilt

nConfigs = len(azimuths)*len(tilts)

weather_path = 'data/solcast_20cities_flanders'
weather_files = os.listdir(weather_path)

Power = 4
n = 100
D = 30

min_size = 10
max_size = 101
size_step = 10

# lp_path = "data/15minuteFluvius.csv"

# lp_df = pd.read_csv(lp_path, sep=';', usecols=['EAN_ID', "Datum_Startuur", "Volume_Afname_kWh", "Volume_Injectie_kWh", "PV-Installatie_Indicator"])

# lp_df.rename(columns={"Datum_Startuur": "Datetime", "Volume_Afname_kWh": "Load", "Volume_Injectie_kWh": "Generation", "PV-Installatie_Indicator": "PV"}, inplace=True)

# lp_df["Load"] = lp_df["Load"] * 4
# lp_df["Generation"] = lp_df["Generation"] * 4

# lp_df["Net"] = lp_df["Load"] - lp_df["Generation"]

# lp_df["Datetime"] = pd.to_datetime(lp_df["Datetime"])

# solarIDs = lp_df[lp_df['PV'] != 0]['EAN_ID'].drop_duplicates().sort_values().reset_index(drop=True)
# nonSolarIDs = lp_df[lp_df['PV'] == 0]['EAN_ID'].drop_duplicates().sort_values().reset_index(drop=True)

lp_df = pickle.load(open("data/lp_df_nonSolar.pkl", 'rb'))

def generate_random_numbers_uniform(n, S):
    random_points = np.sort(np.random.uniform(0, S, n-1))
    random_points = np.concatenate(([0], random_points, [S]))
    random_numbers = np.diff(random_points)
    return random_numbers


def calculate_poa(weather, solar_position, surface_tilt, surface_azimuth):
    poa = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=weather['dni'],
        ghi=weather['ghi'],
        dhi=weather['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        model='isotropic')
    return poa['poa_global']


def add_solar_tau(df):

    location = df['location'].iloc[0]
    times = df.index

    solpos = location.get_solarposition(times)
    elevation = solpos['elevation']

    df = df.copy()
    df['daylight'] = elevation > 0
    df['tau'] = np.nan

    df_day = df[df['daylight']]

    for date, g in df_day.groupby(df_day.index.date):
        tau = (g.index - g.index.min()) / (g.index.max() - g.index.min())
        df.loc[g.index, 'tau'] = tau.values

    return df


def createLoads(lp_df, n, size=50):
    
    dfs = []

    for iter in range(n):

        sample = lp_df['EAN_ID'].drop_duplicates().sample(size, replace=False)

        lps = lp_df[lp_df['EAN_ID'].isin(sample)]

        df = lps.groupby(["Datetime"]).sum()
        df.drop(columns=['EAN_ID', 'PV'], inplace = True)

        df['Date'] = df.index.strftime('%Y-%m-%d')
        df['Time'] = df.index.strftime('%H:%M')      

        dfs.append(df)

    return dfs


def createGeneration(dfs_in, azimuths, tilts, weather_path, weather_files, size=50, Power=4, Pen=0.5):
    dfs = deepcopy(dfs_in)
    for df in dfs:

        weather_file = choice(weather_files)

        weather = pd.read_csv(weather_path + '/' + str(weather_file), usecols=[0,1,2,3])
        weather['period_end'] = pd.to_datetime(weather['period_end'])
        weather.set_index('period_end', inplace=True)

        weather['Date'] = weather.index.strftime('%Y-%m-%d')
        weather['Time'] = weather.index.strftime('%H:%M')

        latitude = float(weather_file[4:9])
        longitude = float(weather_file[10:14])
        location = Location(latitude=latitude, longitude=longitude)

        times = weather.index
        solar_position = location.get_solarposition(times)

        nPanels = max([int(size * Pen), 1])
        arrays = []

        for i in range(nPanels):

            azimuth = uniform(azimuths[0], azimuths[-1])
            tilt = uniform(tilts[0], tilts[-1])
            module_parameters = {'pdc0': Power, 'gamma_pdc': gamma}
            array = pvsystem.Array(pvsystem.FixedMount(tilt, azimuth), module_parameters=module_parameters, temperature_model_parameters=dict(a=-3.56, b=-0.075, deltaT=3))

            arrays.append(array)
        
        system = pvsystem.PVSystem(arrays=arrays, inverter_parameters={'pdc0': Power*nPanels})

        mc = ModelChain(system, location, aoi_model='physical', spectral_model='no_loss')
        mc.run_model(weather)

        PV = mc.results.ac

        df['Generation'] = PV.values[:len(df)]
        df['Net'] = df['Load'] - df['Generation']
        df['ghi'] = weather['ghi'].values[:len(df)]
        df['dni'] = weather['dni'].values[:len(df)]
        df['dhi'] = weather['dhi'].values[:len(df)]
        df['location'] = location
        df['PVkWp'] = nPanels * Power

    return dfs


def extractFeatures(dfs_in, D=30, res=4):

    dfs = deepcopy(dfs_in)
    substation_data_MC = []
    Is = []
    DailyStats_list = []

    for i in range(len(dfs_in)):
        
        dfs[i] = dfs[i][~dfs[i].index.weekday.isin([5, 6])]       # Remove weekends

        df = dfs[i].between_time('05:00', '19:00')                # Filter timesteps outside 5am to 7pm range

        IrrDaily = df.pivot_table(index=['Date'], columns='Time', values='ghi')
        IrrDaily.dropna(inplace=True)

        dfDaily = df.pivot_table(index=['Date'], columns='Time', values='Net')
        dfDaily.dropna(inplace=True)

        IrrDaily = IrrDaily[IrrDaily.index.isin(dfDaily.index)]

        DailyStats = {}

        DailyStats['mean']    = dfDaily.mean(axis=0)                                          # Daily mean
        DailyStats['std']     = dfDaily.std(axis=0)                                           # Daily standard deviation
        DailyStats['min']     = dfDaily.min(axis=0)                                           # Daily minimum
        DailyStats['max']     = dfDaily.max(axis=0)                                           # Daily maximum
        DailyStats['max-min'] = DailyStats['max'] - DailyStats['min']                         # Daily max-min range
        DailyStats['skew']    = dfDaily.skew(axis=0)                                          # Daily skewness
        DailyStats['kurt']    = dfDaily.kurtosis(axis=0)                                      # Daily kurtosis
        DailyStats['corr']    = np.nan_to_num(pearsonr(x=dfDaily, y=IrrDaily, axis=0).statistic, nan=0)             # Correlation between Daily Net Load and Irradiance
        DailyStats['cov']     = np.array([np.cov(dfDaily.iloc[:,i], IrrDaily.iloc[:,i])[0][1] 
                                            for i in range(dfDaily.shape[1])])                # Covariance between Daily Net Load and Irradiance

        # Baseline
        minNetLoad = np.clip(-DailyStats['min'].min(), 0, None)

        # # Daily Features Statistics
        # dfStatsMax = dfDailyStats.max()
        # dfStatsMin = dfDailyStats.min()
        # dfStatsMean = dfDailyStats.mean()
        # dfStatsMedian = dfDailyStats.median()
        # dfStatsStd = dfDailyStats.std()

        # Convert to List
        # subData = pd.concat([dfStatsMax, dfStatsMin, dfStatsMean, dfStatsMedian, dfStatsStd], axis=0).to_list()

        dfDailyStats = pd.DataFrame(DailyStats)

        subData = {}
        for key_main in dfDailyStats.keys():
            for key_sub in dfDailyStats.index:
                subData[f'{key_main}_{key_sub}'] = dfDailyStats[key_main][key_sub]

                
        # Add Label and Baseline
        subData['maxPV'] = df['PVkWp'].iloc[0]
        subData['baseline'] = minNetLoad

        times = dfs[i].index
        solar_position = dfs[i]['location'].iloc[0].get_solarposition(times)

        I = []

        for azimuth in azimuths:
            for tilt in tilts:
                poa = calculate_poa(dfs[i], solar_position, tilt, azimuth)
                I.append(poa)
        I = np.array(I)
        
        Is.append(I)
        substation_data_MC.append(subData)
        DailyStats_list.append(DailyStats)

    return dfs, substation_data_MC, Is, DailyStats_list


def extractFeatures_solar(dfs_in, azimuths, tilts, res=4):

    dfs = deepcopy(dfs_in)
    substation_data_MC = []
    Is = []
    DailyStats_list = []

    tau_bins = np.linspace(0, 1, res + 1)

    for i in range(len(dfs)):
        
        # Remove weekends
        dfs[i] = dfs[i][~dfs[i].index.weekday.isin([5, 6])]

        # Add solar-relative time
        df = add_solar_tau(dfs[i])
        dfs[i]['tau'] = df['tau']
        df = df[df['daylight']]

        # Assign solar bins
        df['tau_bin'] = pd.cut(df['tau'], bins=tau_bins, labels=False, include_lowest=True)

        # Pivot by solar bin instead of clock time
        NetDaily = df.pivot_table(index=df.index.date, columns='tau_bin', values='Net')
        NetDaily.dropna(inplace=True)

        ghi = df['ghi']
        scale = np.nanpercentile(ghi, 99)
        df['ghi_norm'] = ghi / scale

        IrrDaily = df.pivot_table(index=df.index.date, columns='tau_bin', values='ghi_norm')
        IrrDaily = IrrDaily.loc[NetDaily.index]

        DailyStats = {}

        DailyStats['mean']    = NetDaily.mean(axis=0)
        DailyStats['std']     = NetDaily.std(axis=0)
        DailyStats['min']     = NetDaily.min(axis=0)
        DailyStats['max']     = NetDaily.max(axis=0)
        DailyStats['max-min'] = DailyStats['max'] - DailyStats['min']
        DailyStats['skew']    = NetDaily.skew(axis=0)
        DailyStats['kurt']    = NetDaily.kurtosis(axis=0)
        DailyStats['corr']    = np.nan_to_num(
            pearsonr(x=NetDaily, y=IrrDaily, axis=0).statistic, nan=0
        )
        DailyStats['cov']     = np.array([
            np.cov(NetDaily.iloc[:, j], IrrDaily.iloc[:, j])[0][1]
            for j in range(NetDaily.shape[1])
        ])

        # Baseline
        minNetLoad = np.clip(-DailyStats['min'].min(), 0, None)

        dfDailyStats = pd.DataFrame(DailyStats)

        subData = {}
        for stat in dfDailyStats.columns:
            for b in dfDailyStats.index:
                subData[f'{stat}_tau{b}'] = dfDailyStats.loc[b, stat]

        subData['maxPV'] = df['PVkWp'].iloc[0]
        subData['baseline'] = minNetLoad

        # POA irradiance (unchanged)
        times = dfs[i].index
        solar_position = dfs[i]['location'].iloc[0].get_solarposition(times)

        I = []
        for azimuth in azimuths:
            for tilt in tilts:
                poa = calculate_poa(dfs[i], solar_position, tilt, azimuth)
                I.append(poa)

        Is.append(np.array(I))
        substation_data_MC.append(subData)
        DailyStats_list.append(DailyStats)
        

    return dfs, substation_data_MC, Is, DailyStats_list


def createSubstationData(size=50, lp_df=lp_df, Power=Power, n=1, weather_files=weather_files, weather_path=weather_path, azimuths=azimuths, tilts=tilts, Pen=0.5):
    substation_data_MC = []
    sizes_MC = []
    dfs = []
    Is = []
    DailyStats = []
    
    for iter in trange(n):
        
        sizes_MC.append(size)

        df = createLoads(lp_df, 1, size=size)
        df = createGeneration(df, azimuths, tilts, weather_path, weather_files, size=size, Pen=Pen, Power=Power)
        df, substation_data, I, DailyStat = extractFeatures_solar(df, azimuths, tilts)
        
        dfs.extend(df)
        substation_data_MC.extend(substation_data)
        Is.extend(I)
        DailyStats.extend(DailyStat)

        substation_data_Df_MC = pd.DataFrame(data=substation_data_MC, columns=substation_data_MC[0].keys())

    return substation_data_Df_MC, sizes_MC, dfs, Is


def loadDataFluvius():

    substation_data_Df_MC = pickle.load(open('data/substationDf_Irr_MC.pkl', 'rb'))
    sizes_MC = pickle.load(open('data/sizes_MC.pkl', 'rb'))
    dfs = pickle.load(open('data/dfs.pkl', 'rb'))
    Is = pickle.load(open('data/Is.pkl', 'rb'))
    

    return substation_data_Df_MC, sizes_MC, dfs, Is


def visualizeSubstationLoad(dfs, index=0, date='2022-01-03', vmin=None, vmax=None):
    try:
        df = dfs[index]
        # df['tau'] = df['tau'].fillna(0)
    except IndexError:
        print("Index out of bounds")
        return

    if date not in df['Date'].values:
        print("Date not found in the data for this substation")
        return
    else:
        df_date = df[df.index.date == pd.to_datetime(date).date()]


    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(df_date.index, df_date['Net'], label='Net Load', color='blue')
    ax2.plot(df_date.index, df_date['ghi'], label='GHI', color='orange')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Net Load', color='blue')
    ax2.set_ylabel('GHI', color='orange')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='orange')
    plt.title(f'Substation {index} Load and GHI on {date}, PV Capacity: {df_date["PVkWp"].iloc[0]:.2f} kWp')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df_year = dfs[index]
    df_year_pivot = df_year.pivot_table(index=df_year.index.date, columns=df_year.index.strftime('%H:%M'), values='Net')

    plt.figure(figsize=(14, 8))
    plt.imshow(df_year_pivot.values, aspect='auto', cmap='RdYlGn_r', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Net Load (kW)')
    plt.xlabel('Time of Day')
    plt.ylabel('Date')
    plt.title(f'Substation {index} Net Load Heatmap - Year View')
    plt.xticks(np.arange(0, len(df_year_pivot.columns), 4), df_year_pivot.columns[::4], rotation=45)
    plt.yticks(np.arange(0, len(df_year_pivot.index), 30), df_year_pivot.index[::30], rotation=0)
    plt.tight_layout()
    plt.show()


    return


def visualizeSubstationLoadSolarTime(dfs, index=0, date='2022-01-03', res=10, vmin=None, vmax=None):
    try:
        df = dfs[index]
        df = df[df['tau'].notnull()]
    except IndexError:
        print("Index out of bounds")
        return

    if date not in df['Date'].values:
        print("Date not found in the data for this substation")
        return
    else:
        df_date = df[df.index.date == pd.to_datetime(date).date()]


    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(df_date.tau, df_date['Net'], label='Net Load', color='blue')
    ax2.plot(df_date.tau, df_date['ghi'], label='GHI', color='orange')
    ax1.set_xlabel('Relative Solar Time')
    ax1.set_ylabel('Net Load', color='blue')
    ax2.set_ylabel('GHI', color='orange')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='orange')
    plt.title(f'Substation {index} Load and GHI on {date}, PV Capacity: {df_date["PVkWp"].iloc[0]:.2f} kWp')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df_year = dfs[index]

    tau_bins = np.linspace(0, 1, res + 1)

    df_year['tau_bin'] = pd.cut(df_year['tau'], bins=tau_bins, labels=False, include_lowest=True)
    df_year_pivot = df_year.pivot_table(index=df_year.index.date, columns=df_year.tau_bin, values='Net')

    plt.figure(figsize=(14, 8))
    plt.imshow(df_year_pivot.values, aspect='auto', cmap='RdYlGn_r', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Net Load (kW)')
    plt.xlabel('Relative Solar Time Bin')
    plt.ylabel('Date')
    plt.title(f'Substation {index} Net Load Heatmap - Year View')
    plt.xticks(np.arange(0, len(df_year_pivot.columns), 1), df_year_pivot.columns[::1], rotation=45)
    plt.yticks(np.arange(0, len(df_year_pivot.index), 30), df_year_pivot.index[::30], rotation=0)
    plt.tight_layout()
    plt.show()


    return


def splitData(substation_data_Df_MC, dfs, Is, test_size=1):

    if test_size == 1 or test_size == 0:

        scaler = pickle.load(open('models/scaler_MC.pkl', 'rb'))

        return {
            "dfs_Train": dfs,
            "dfs_Test": dfs,
            "Is_Train": Is,
            "Is_Test": Is,
            "XIrrTrain_MC": substation_data_Df_MC.drop(columns=['maxPV', 'baseline']),
            "XIrrTrain_MC_normalized": scaler.transform(substation_data_Df_MC.drop(columns=['maxPV', 'baseline'])),
            "yIrrTrain_MC": substation_data_Df_MC['maxPV'],
            "XIrrTest_MC": substation_data_Df_MC.drop(columns=['maxPV', 'baseline']),
            "XIrrTest_MC_normalized": scaler.transform(substation_data_Df_MC.drop(columns=['maxPV', 'baseline'])),
            "yIrrTest_MC": substation_data_Df_MC['maxPV'],
            "scaler": scaler
        }

    test_slice = int(len(substation_data_Df_MC) * (1 - test_size))

    substation_data_Df_MC_Train = substation_data_Df_MC.iloc[:test_slice]
    substation_data_Df_MC_Test  = substation_data_Df_MC.iloc[test_slice:]

    dfs_Train = dfs[:test_slice]
    dfs_Test  = dfs[test_slice:]

    Is_Train = Is[:test_slice]
    Is_Test  = Is[test_slice:]

    XIrrTrain_MC = substation_data_Df_MC_Train.drop(columns=['maxPV', 'baseline'])
    yIrrTrain_MC = substation_data_Df_MC_Train['maxPV']

    XIrrTest_MC = substation_data_Df_MC_Test.drop(columns=['maxPV', 'baseline'])
    yIrrTest_MC = substation_data_Df_MC_Test['maxPV']

    XIrrTrain_MC = XIrrTrain_MC.dropna(axis=1)
    XIrrTest_MC = XIrrTest_MC.dropna(axis=1)

    scaler_MC = StandardScaler()

    XIrrTrain_MC_normalized = scaler_MC.fit_transform(XIrrTrain_MC)
    XIrrTest_MC_normalized = scaler_MC.transform(XIrrTest_MC)

    return_dict = {
        "dfs_Train": dfs_Train,
        "dfs_Test": dfs_Test,
        "Is_Train": Is_Train,
        "Is_Test": Is_Test,
        "XIrrTrain_MC": XIrrTrain_MC,
        "XIrrTrain_MC_normalized": XIrrTrain_MC_normalized,
        "yIrrTrain_MC": yIrrTrain_MC,
        "XIrrTest_MC": XIrrTest_MC,
        "XIrrTest_MC_normalized": XIrrTest_MC_normalized,
        "yIrrTest_MC": yIrrTest_MC,
        "scaler": scaler_MC
    }

    return return_dict


def viewFeatures(X, y):
    for column in X.columns:
        plt.figure()
        plt.scatter(X[column], y, marker='o')
        plt.xlabel(column)
        plt.ylabel('Installed PV [kWp]')
        plt.title(f'Scatter plot of {column} vs kWp')
        plt.show()
    return


def viewFeatureCorrelations(X, y):

    corr_train = pd.concat([X, y], axis=1).corr()

    corr_train_rounded = corr_train.round(3)

    styled_corr_train = corr_train_rounded.style.background_gradient(cmap='coolwarm', axis=None, gmap=corr_train_rounded.abs())
    styled_corr_train = styled_corr_train.format(precision=3)

    display(styled_corr_train)
    return


def tune_and_train_xgb(XIrrTrain_MC, yIrrTrain_MC):

    def objectiveMC(space):
        reg=xgb.XGBRegressor(learning_rate = space['eta'], n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'],
                            subsample = space['subsample'], reg_alpha = space['reg_alpha'], reg_lambda = space['reg_lambda'], 
                            min_child_weight = int(space['min_child_weight']), colsample_bytree = space['colsample_bytree'], eval_metric = "mae")
        
        accuracy = -np.mean(cross_val_score(reg, XIrrTrain_MC, yIrrTrain_MC, scoring='neg_mean_squared_error', cv=5))
    
    
        return {'loss': accuracy, 'status': STATUS_OK }

    space={ 'eta': hp.uniform('eta', 0, 1),                                     # Learning Rate
        'max_depth': hp.quniform("max_depth", 3, 18, 1),                        # Maximum Tree depth
        'gamma': hp.quniform ('gamma', 0, 100, 1),                                  # Pruning parameter (node gain < gamma -> Prune)
        'subsample': hp.uniform('subsample', 0, 1),                             # Fraction of observations to be randomly sampled for each tree
        'reg_alpha' : hp.uniform('reg_alpha', 0, 1),                            # L1-norm (Lasso) Regularization 
        'reg_lambda' : hp.quniform('reg_lambda', 0, 100, 1),                        # L2-norm (Ridge) Regularization 
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.1, 1),            # Ratio of randomly sampled features to use for each tree
        'min_child_weight' : hp.quniform('min_child_weight', 0, 100, 1),        # Minimum sum of weights of all observations required in a child (in regression: weight = 1)
        'n_estimators': hp.quniform('n_estimators', 50, 1000, 10),               # Number of trees
    }

    trials = Trials()

    start_time_ht = time.time()
    best_hyperparams_MC = fmin(fn = objectiveMC,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 1000,
                            trials = trials)
    end_time_ht = time.time()
    print("Hyperparameter tuning took " + str(end_time_ht - start_time_ht) + " seconds")

    xgb_model_MC = xgb.XGBRegressor(
        learning_rate=best_hyperparams_MC['eta'],
        n_estimators=int(best_hyperparams_MC['n_estimators']),
        max_depth=int(best_hyperparams_MC['max_depth']),
        gamma=best_hyperparams_MC['gamma'],
        subsample=best_hyperparams_MC['subsample'],
        reg_alpha=best_hyperparams_MC['reg_alpha'],
        reg_lambda=best_hyperparams_MC['reg_lambda'],
        min_child_weight=int(best_hyperparams_MC['min_child_weight']),
        colsample_bytree=best_hyperparams_MC['colsample_bytree'],
        eval_metric="mae"
    )

    start_time_fit = time.time()
    xgb_model_MC.fit(XIrrTrain_MC, yIrrTrain_MC)
    end_time_fit = time.time()
    
    print("Fitting took " + str(end_time_fit - start_time_fit) + " seconds")
    return xgb_model_MC, best_hyperparams_MC


def tune_and_train_lasso(XIrrTrain_MC, yIrrTrain_MC):
    lasso = Lasso()
    
    space = {
        'alpha': hp.loguniform('alpha', -5, 1)
    }

    trials = Trials()

    start_time_ht = time.time()
    best_hyperparams_lasso = fmin(fn=lambda params: -np.mean(cross_val_score(Lasso(**params), XIrrTrain_MC, yIrrTrain_MC, scoring='neg_mean_squared_error', cv=5)),
                                space=space,
                                algo=tpe.suggest,
                                max_evals=1000,
                                trials=trials)
    end_time_ht = time.time()
    print("Hyperparameter tuning took " + str(end_time_ht - start_time_ht) + " seconds")

    lasso.set_params(**best_hyperparams_lasso)

    start_time_fit = time.time()
    lasso.fit(XIrrTrain_MC, yIrrTrain_MC)
    end_time_fit = time.time()
    print("Fitting took " + str(end_time_fit - start_time_fit) + " seconds")
    
    return lasso


def tune_and_train_elast_net(XIrrTrain_MC, yIrrTrain_MC):
    
    def objective(space):
        model = ElasticNet(alpha=space['alpha'], l1_ratio=space['l1_ratio'])
        mse = -np.mean(cross_val_score(model, XIrrTrain_MC, yIrrTrain_MC, scoring='neg_mean_squared_error', cv=5))
        return {'loss': mse, 'status': STATUS_OK}

    # Initialize ElasticNet model
    elast_net = ElasticNet()

    space = {
        'alpha': hp.loguniform('alpha', -5, 1),
        'l1_ratio': hp.uniform('l1_ratio', 0.01, 1)
    }

    trials = Trials()

    start_time_ht = time.time()
    best_hyperparams_en = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=1000,
                            trials=trials)
    end_time_ht = time.time()
    print("Hyperparameter tuning took " + str(end_time_ht - start_time_ht) + " seconds")

    elast_net.set_params(**best_hyperparams_en)

    start_time_fit = time.time()
    elast_net.fit(XIrrTrain_MC, yIrrTrain_MC)
    end_time_fit = time.time()
    print("Fitting took " + str(end_time_fit - start_time_fit) + " seconds")

    return elast_net


def tune_and_train_svr(XIrrTrain_MC, yIrrTrain_MC):
    svr = SVR()

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 1, 10],
        'degree': [2, 3, 4],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }

    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)

    start_time_fit = time.time()
    grid_search.fit(XIrrTrain_MC, yIrrTrain_MC)
    end_time_fit = time.time()
    print("Grid search fitting took " + str(end_time_fit - start_time_fit) + " seconds")

    svr = grid_search.best_estimator_

    return svr


def create_ffnn(
    neurons=32,
    dropout_rate=0,
    learning_rate=0.05,
    kr_l1=1e-4,
    kr_l2=1e-4,
    n_features=1
):

    model = Sequential()
    model.add(Dense(
        neurons,
        activation='relu',
        input_shape=(n_features,),
        kernel_regularizer=regularizers.l1_l2(l1=kr_l1, l2=kr_l2)
    ))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model


def tune_and_train_ffnn(X, y, max_evals=200):

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True
    )

    def objective(params):

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        mse_folds = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = create_ffnn(
                neurons=int(params['neurons']),
                learning_rate=params['learning_rate'],
                kr_l1=params['kr_l1'],
                kr_l2=params['kr_l2'],
                n_features=X.shape[1]
            )

            model.fit(
                X_train, y_train,
                epochs=200,
                batch_size=64,
                callbacks=[early_stop],
                verbose=0
            )

            mse = model.evaluate(X_test, y_test, verbose=0)
            mse_folds.append(mse)

        return {
            'loss': np.mean(mse_folds),
            'status': STATUS_OK
        }

    space = {
    'neurons': hp.quniform('neurons', 64, 160, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.02), np.log(0.2)),
    'kr_l1': hp.loguniform('kr_l1', np.log(1e-7), np.log(5e-5)),
    'kr_l2': hp.loguniform('kr_l2', np.log(1e-7), np.log(5e-5))
}

    trials = Trials()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

    # Train final model on full dataset
    best_model = create_ffnn(
        neurons=int(best['neurons']),
        learning_rate=best['learning_rate'],
        kr_l1=best['kr_l1'],
        kr_l2=best['kr_l2'],
        n_features=X.shape[1]
    )

    history = best_model.fit(
        X, y,
        epochs=200,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )

    return best_model, history, best, trials


def train_all(X, y):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    xgb_model = tune_and_train_xgb(X, y)
    lin_model = LinearRegression()
    lin_model.fit(X_scaled, y)
    lasso_model = tune_and_train_lasso(X_scaled, y)
    elast_model = tune_and_train_elast_net(X_scaled, y)
    svr_model = tune_and_train_svr(X_scaled, y)
    ffnn_model, history, best, trials = tune_and_train_ffnn(X_scaled, y)

    return xgb_model, lin_model, lasso_model, elast_model, svr_model, ffnn_model


def loadTrainedModels(models=["Support Vector Regression"]):

    model_dict = {}
    if "XGBoost" in models:
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model("models/xgb_model_MC.json")
        model_dict["XGBoost"] = xgb_model

    if "Linear Regression" in models:
        lin_model = pickle.load(open("models/lin_model_MC.json", 'rb'))
        model_dict["Linear Regression"] = lin_model

    if "Lasso Regression" in models:
        lasso_model = pickle.load(open("models/lasso_model_MC.json", 'rb'))
        model_dict["Lasso Regression"] = lasso_model

    if "Elastic Net" in models:
        elast_model = pickle.load(open("models/elast_model_MC.json", 'rb'))
        model_dict["Elastic Net"] = elast_model

    if "Support Vector Regression" in models:
        svr_model = pickle.load(open("models/svr_model_MC.json", 'rb'))
        model_dict["Support Vector Regression"] = svr_model

    if "Feedforward Neural Network" in models:
        ffnn_model = keras.models.load_model("models/ffnn_model_MC.h5")
        model_dict["Feedforward Neural Network"] = ffnn_model

    return model_dict


def method_a_optimization(netLoads, Is, tilts, azimuths, verbose=True):
    methodA = []
    J = len(tilts) * len(azimuths)
    n = len(netLoads)

    if verbose:
        start_time_pred = time.time()
        for iter in trange(n, desc="Method A Optimization"): 
            netLoad = netLoads[iter]['Net'].values
            I = Is[iter] / 1000

            # Decision variable
            x = cp.Variable(J)  # x is the installed kWp of PV

            # Objective function: Sum of the terms
            objectiveA = cp.Minimize(
                cp.norm(cp.diff(netLoad) + cp.diff(x.T @ I), 1)
            )

            # Constraints
            constraints = [x >= 0]

            # Define and solve the optimization problem
            problemA = cp.Problem(objectiveA, constraints)
            problemA.solve(verbose=False)

            methodA.append(sum(x.value))

        end_time_pred = time.time()
        print("Prediction took " + str(end_time_pred - start_time_pred) + " seconds")
    else:
        for iter in range(n):
            netLoad = netLoads[iter]['Net'].values
            I = Is[iter] / 1000

            # Decision variable
            x = cp.Variable(J)  # x is the installed kWp of PV

            # Objective function: Sum of the terms
            objectiveA = cp.Minimize(
                cp.norm(cp.diff(netLoad) + cp.diff(x.T @ I), 1)
            )

            # Constraints
            constraints = [x >= 0]

            # Define and solve the optimization problem
            problemA = cp.Problem(objectiveA, constraints)
            problemA.solve(verbose=False)

            methodA.append(sum(x.value))
    return methodA


def method_b_optimization(netLoads, Is, tilts, azimuths, lambda_reg=1, verbose=True):
    methodB = []
    J = len(tilts) * len(azimuths)
    n = len(netLoads)

    if verbose:
        start_time_pred = time.time()
        for iter in trange(n, desc="Method B Optimization"):
            netLoad = netLoads[iter]['Net'].values
            I = Is[iter] / 1000

            K = len(netLoad)

            # Decision variable
            x = cp.Variable(J)  # x is the installed kWp of PV
            y = cp.Variable(K)  # y is the Load

            # Objective function: Sum of the terms
            objectiveB = cp.Minimize(
                cp.sum_squares(netLoad - (y - x.T @ I)) +
                lambda_reg * cp.norm(cp.diff(y), 1)
            )

            # Constraints
            constraints = [x >= 0,
                        y >= 0]

            # Define and solve the optimization problem
            problemB = cp.Problem(objectiveB, constraints)
            try:
                problemB.solve(solver=cp.CLARABEL, verbose=False)
                methodB.append(sum(x.value))
            except:
                try:
                    problemB.solve(solver=cp.SCS, verbose=False)
                    methodB.append(sum(x.value))
                except:
                    try:
                        problemB.solve(solver=cp.ECOS, verbose=False)
                        methodB.append(sum(x.value))
                    except:
                        print("No solution found for iteration {}".format(iter))
                        methodB.append(0)
                

        end_time_pred = time.time()
        print("Prediction took " + str(end_time_pred - start_time_pred) + " seconds")

    else:
        for iter in range(n):
            netLoad = netLoads[iter]['Net'].values
            I = Is[iter] / 1000

            K = len(netLoad)

            # Decision variable
            x = cp.Variable(J)  # x is the installed kWp of PV
            y = cp.Variable(K)  # y is the Load

            # Objective function: Sum of the terms
            objectiveB = cp.Minimize(
                cp.sum_squares(netLoad - (y - x.T @ I)) +
                lambda_reg * cp.norm(cp.diff(y), 1)
            )

            # Constraints
            constraints = [x >= 0,
                        y >= 0]

            # Define and solve the optimization problem
            problemB = cp.Problem(objectiveB, constraints)
            try:
                problemB.solve(solver=cp.CLARABEL, verbose=False)
                methodB.append(sum(x.value))
            except:
                try:
                    problemB.solve(solver=cp.SCS, verbose=False)
                    methodB.append(sum(x.value))
                except:
                    try:
                        problemB.solve(solver=cp.ECOS, verbose=False)
                        methodB.append(sum(x.value))
                    except:
                        print("No solution found for iteration {}".format(iter))
                        methodB.append(0)

    return methodB


def method_c_optimization(netLoads, Is, tilts, azimuths, c=5, verbose=True):
    methodC = []
    J = len(tilts) * len(azimuths)
    n = len(netLoads)
    Loads = []


    if verbose:
        start_time_pred = time.time()
        for iter in trange(n, desc="Method C Optimization"): 
            netLoad = netLoads[iter]['Net'].values
            I = Is[iter] / 1000
            K = len(netLoad)

            # Decision variable
            x = cp.Variable(J)  # x is the installed kWp of PV
            y = cp.Variable(K)  # y is the Load

            # Objective function: Sum of the terms
            objectiveC = cp.Minimize(
                cp.sum_squares(netLoad - (y - x.T @ I))
            )

            # Constraints
            constraints = []
            # for i in range(0, K, c):
            #     for j in range(1, c):
            #         constraints += [
            #             y[i] - y[i + j] == 0  # Load is piecewise constant in lengths of c
            #         ]

            mask = np.ones(K-1, dtype=bool)
            mask[np.arange(c-1, K-1, c)] = False

            constraints += [
                cp.diff(y)[mask] == 0,
                x >= 0,
                y >= 0
            ]

            # Define and solve the optimization problem
            problemC = cp.Problem(objectiveC, constraints)
            problemC.solve(solver=cp.CLARABEL, verbose=False)

            methodC.append(sum(x.value))
            Loads.append(y)

        end_time_pred = time.time()
        print("Prediction took " + str(end_time_pred - start_time_pred) + " seconds")
    else:
        for iter in range(n):
            netLoad = netLoads[iter]['Net'].values
            I = Is[iter] / 1000
            K = len(netLoad)

            # Decision variable
            x = cp.Variable(J)  # x is the installed kWp of PV
            y = cp.Variable(K)  # y is the Load

            # Objective function: Sum of the terms
            objectiveC = cp.Minimize(
                cp.sum_squares(netLoad - (y - x.T @ I))
            )

            # Constraints
            constraints = []
            # for i in range(0, K, c):
            #     for j in range(1, c):
            #         constraints += [
            #             y[i] - y[i + j] == 0  # Load is piecewise constant in lengths of c
            #         ]

            mask = np.ones(K-1, dtype=bool)
            mask[np.arange(c-1, K-1, c)] = False

            constraints += [
                cp.diff(y)[mask] == 0,
                x >= 0,
                y >= 0
            ]

            # Define and solve the optimization problem
            problemC = cp.Problem(objectiveC, constraints)
            problemC.solve(solver=cp.CLARABEL, verbose=False)

            methodC.append(sum(x.value))
            Loads.append(y)
    return methodC, Loads


def calculate_methodINM(dfs, verbose=True):
    methodINM = []
    start_time_pred = time.time()

    if verbose:
        start_time_pred = time.time()
        for df in tqdm(dfs, desc="Irradiance GHI Calculation"):

            # df = df[df['ghi'] > 0]
            X = df['ghi'].values.reshape(-1, 1)/1000
            y = df['Net'].values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(X,y)

            methodINM.append(np.abs(model.coef_[0][0]))
        end_time_pred = time.time()
        print("Prediction took " + str(end_time_pred - start_time_pred) + " seconds")
    else:
        for df in dfs:

            # df = df[df['ghi'] > 0]
            X = df['ghi'].values.reshape(-1, 1)/1000
            y = df['Net'].values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(X,y)

            methodINM.append(np.abs(model.coef_[0][0]))
    return methodINM


def calculate_methodINM_POA(dfs, Is, alpha=28, verbose=True):
    methodINM = []

    if verbose:
        start_time_pred = time.time()
        for i in trange(len(dfs), desc="Irradiance POA Calculation"):

            df = dfs[i][dfs[i]['ghi'] > 0]
            I = Is[i]
            
            X = I.T[dfs[i]['ghi'] > 0] / 1000
            y = df['Net'].values.reshape(-1, 1)

            model = Ridge(alpha=alpha)
            model.fit(X,y)

            methodINM.append(np.abs(model.coef_.sum()))

        end_time_pred = time.time()
        print("Prediction took " + str(end_time_pred - start_time_pred) + " seconds")
    else:
        for i in range(len(dfs)):

            df = dfs[i][dfs[i]['ghi'] > 0]
            I = Is[i]
            
            X = I.T[dfs[i]['ghi'] > 0] / 1000
            y = df['Net'].values.reshape(-1, 1)

            model = Ridge(alpha=alpha)
            model.fit(X,y)

            methodINM.append(np.abs(model.coef_.sum()))
            
    return methodINM


def modelBasedPrections(dfs, Is, load_estimation=True, tilts=tilts, azimuths=azimuths, methods_in=['Method INM (POA)', "Method B"]):

    methods = {}

    if load_estimation:
        if "Method A" in methods_in:
            methodA = pickle.load(open("data/methodA.pkl", 'rb'))
            methods["Method A"] = methodA
        if "Method B" in methods_in:
            methodB = pickle.load(open("data/methodB.pkl", 'rb'))
            methods["Method B"] = methodB
        if "Method C" in methods_in:
            methodC,_ = pickle.load(open("data/methodC.pkl", 'rb'))
            methods["Method C"] = methodC
        if "Method INM (GHI)" in methods_in:
            methodINM = pickle.load(open("data/methodINM.pkl", 'rb'))
            methods["Method INM (GHI)"] = methodINM
        if "Method INM (POA)" in methods_in:
            methodINM_POA = pickle.load(open("data/methodINM_POA.pkl", 'rb'))
            methods["Method INM (POA)"] = methodINM_POA

    else:
        if "Method A" in methods_in:
            methodA = method_a_optimization(dfs, Is, tilts, azimuths, verbose=False)
            methods["Method A"] = methodA
        if "Method B" in methods_in:
            methodB = method_b_optimization(dfs, Is, tilts, azimuths, verbose=False)
            methods["Method B"] = methodB
        if "Method C" in methods_in:
            methodC,_ = method_c_optimization(dfs, Is, tilts, azimuths, verbose=False)
            methods["Method C"] = methodC
        if "Method INM (GHI)" in methods_in:
            methodINM = calculate_methodINM(dfs, verbose=False)
            methods["Method INM (GHI)"] = methodINM
        if "Method INM (POA)" in methods_in:
            methodINM_POA = calculate_methodINM_POA(dfs, Is, verbose=False)
            methods["Method INM (POA)"] = methodINM_POA

    return methods


def predict_and_plot(XTest_scaled, yTest, models, methods):

    for model_name, model in models.items():
        print(f"Predicting with {model_name}...")
        y_pred = model.predict(XTest_scaled)
        rmse = root_mean_squared_error(yTest.values, y_pred)
        
        fig = plt.figure()
        plt.scatter(yTest.values, y_pred, marker='+')
        plt.plot(yTest.values,yTest.values, ls = ':', color='r')
        plt.xlabel('Real PV Capacity [kWp]')
        plt.ylabel('Estimated PV Capacity [kWp]')
        plt.legend([f"{model_name} Model", 'Ground Truth'])
        plt.annotate("RMSE = {:.3f}".format(rmse), (0, 200))
        plt.show()

    for method_name, method_pred in methods.items():
        print(f"Predicting with {method_name}...")
        rmse = root_mean_squared_error(yTest.values, method_pred)
        
        fig = plt.figure()
        plt.scatter(yTest.values, method_pred, marker='+')
        plt.plot(yTest.values,yTest.values, ls = ':', color='r')
        plt.xlabel('Real PV Capacity [kWp]')
        plt.ylabel('Estimated PV Capacity [kWp]')
        plt.legend([f"{method_name}", 'Ground Truth'])
        plt.annotate("RMSE = {:.3f}".format(rmse), (0, 200))
        plt.show()

    return


def loadAusGridData():
    scaler_30min = pickle.load(open("models/scaler_30min.pkl", 'rb'))
    ausgrid_file = 'data/ausgrid/2012-2013 Solar home electricity data v2.csv'

    Postcodes = [2259, 2261, 2290]

    Postcode_weather_files = {
        2259: 'csv_-33.231623_151.406736_2259.csv',
        2261: 'csv_-33.356579_151.486092_2261.csv',
        2290: 'csv_-32.967883_151.694587_2290.csv'
    }

    raw_data_aus = pd.read_csv(ausgrid_file, skiprows=1,
                        parse_dates=['date'], dayfirst=True,
                        na_filter=False, dtype={'Row Quality': str})

    raw_data_aus = raw_data_aus[raw_data_aus['Postcode'].isin(Postcodes)]

    raw_data_aus.sort_values(by=['date'], inplace=True)

    raw_data_aus.date.min(), raw_data_aus.date.max()

    n = 1  # Number of substations per postcode
    aus_pen = 1

    scale = 1

    aus_index = pd.date_range(raw_data_aus.date.min(), raw_data_aus.date.max() + pd.Timedelta(days=1), freq='30T', tz='Australia/Sydney')

    aus_dfs = {}

    for postcode in Postcodes:
        
        aus_dfs[postcode] = []

        df_postcode = raw_data_aus[raw_data_aus['Postcode'] == postcode]
        customers_unique = df_postcode['Customer'].unique()

        for customer in customers_unique.copy():
            df_customer = df_postcode[df_postcode['Customer'] == customer]
            load = df_customer[df_customer['Consumption Category'] == 'GC'].drop(columns=['date', 'Row Quality', 'Postcode', 'Customer', 'Consumption Category', 'Generator Capacity']).values.ravel() * 2

            if len(load) != len(aus_index) -1:
                # print(f"Customer {customer} in Postcode {postcode} has data length {len(load)}, expected {len(aus_index)-1}")
                customers_unique = customers_unique[customers_unique != customer]

        for i in range(n):

            aus_dfs[postcode].append(pd.DataFrame(index=aus_index[:-1], columns=['Net', 'ghi', 'dhi', 'dni', 'PVkWp', 'location']))

            aus_dfs[postcode][i]['Net'].values[:] = 0
            aus_dfs[postcode][i]['Net'] = aus_dfs[postcode][i]['Net'].astype(float)

            aus_dfs[postcode][i]['PVkWp'].values[:] = 0
            aus_dfs[postcode][i]['PVkWp'] = aus_dfs[postcode][i]['PVkWp'].astype(float)
            
            
            selected_customers = customers_unique
            selected_pv_customers = np.random.choice(selected_customers, size=int(len(selected_customers) * aus_pen), replace=False)
            for customer in selected_customers:
                # print(f"Processing Postcode {postcode}, Customer {customer}")
                df_customer = df_postcode[df_postcode['Customer'] == customer]
                
                load = df_customer[df_customer['Consumption Category'] == 'GC'].drop(columns=['date', 'Row Quality', 'Postcode', 'Customer', 'Consumption Category', 'Generator Capacity'])
                load = load.values.ravel() * 2

                if customer in selected_pv_customers:
                    gen = df_customer[df_customer['Consumption Category'] == 'GG'].drop(columns=['date', 'Row Quality', 'Postcode', 'Customer', 'Consumption Category', 'Generator Capacity'])
                    gen = gen.values.ravel() * 2
                    net = load - gen

                    aus_dfs[postcode][i]['Net'] += net * scale
                    pv_capacity = df_customer['Generator Capacity'].values[0]
                    aus_dfs[postcode][i]['PVkWp'] += pv_capacity * scale
                else:
                    aus_dfs[postcode][i]['Net'] += load

        weather_file = Postcode_weather_files[postcode]
        weather_data = pd.read_csv(f'data/ausgrid/{weather_file}', parse_dates=['period_end'])
        weather_data['period_end'] = pd.to_datetime(weather_data['period_end'])
        weather_data.set_index('period_end', inplace=True)
        weather_data = weather_data.tz_convert('Australia/Sydney')
        weather_data.index = weather_data.index.shift(periods=-1, freq='30T')

        weather_data = weather_data[(weather_data.index.date >= raw_data_aus.date.min().date()) & (weather_data.index.date <= raw_data_aus.date.max().date())]

        latitude = float(weather_file[4:14])
        longitude = float(weather_file[15:25])
        location = Location(latitude=latitude, longitude=longitude, tz='Australia/Sydney', altitude=0, name=str(postcode))

        for i in range(n):
            aus_dfs[postcode][i]['ghi'] = weather_data['ghi'].values
            aus_dfs[postcode][i]['dhi'] = weather_data['dhi'].values
            aus_dfs[postcode][i]['dni'] = weather_data['dni'].values

            aus_dfs[postcode][i]['Date'] = aus_dfs[postcode][i].index.strftime('%Y-%m-%d')
            aus_dfs[postcode][i]['Time'] = aus_dfs[postcode][i].index.strftime('%H:%M')

            aus_dfs[postcode][i]['location'] = location

            aus_dfs[postcode][i]


    features_postcodes = {}
    Is_postcodes = {}
    real_values_postcodes = {}
    baseline_postcodes = {}

    azimuths_aus = np.append(np.linspace(270, 330, 3), np.linspace(0, 90, 4))

    for postcode in Postcodes:

        aus_dfs[postcode], substation_data_postcode, Is_postcode, _ = extractFeatures_solar(aus_dfs[postcode], azimuths=azimuths_aus, tilts=tilts)
        
        substation_data_postcode = pd.DataFrame(substation_data_postcode, columns=substation_data_postcode[0].keys())
        features_postcodes[postcode] = substation_data_postcode.drop(columns=['maxPV', 'baseline'])
        features_postcodes[postcode] = scaler_30min.transform(features_postcodes[postcode].dropna(axis=1))
        Is_postcodes[postcode] = Is_postcode
        real_values_postcodes[postcode] = substation_data_postcode['maxPV']
        baseline_postcodes[postcode] = substation_data_postcode['baseline']

    return aus_dfs, features_postcodes, Is_postcodes, real_values_postcodes


def loadTexasData(scaler=None):
    if scaler == None:
        scaler = pickle.load(open("models/scaler_MC.pkl", 'rb'))

    metadata = pd.read_csv('data/15minute_data_austin/metadata.csv', engine='python', encoding="ISO-8859-1", skiprows=[1])

    dataids = metadata[#metadata.active_record.eq('yes') &
                    metadata.city.eq('Austin') &
                    #metadata.egauge_1min_data_availability.isin(['100%', '99%', '98%', '97%']) &
                    metadata.grid.eq('yes')]


    all_data = pd.read_csv('data/15minute_data_austin/15minute_data_austin.csv', engine='python', encoding="ISO-8859-1",
                                    parse_dates=['local_15min'], index_col=['local_15min'])
    all_data = all_data[['dataid', 'grid', 'solar', 'solar2']]

    filt = all_data[all_data.dataid.isin(dataids.dataid)]
    filt.index = pd.to_datetime(filt.index, utc=True, yearfirst=True)
    filt = filt.tz_convert('US/Central')

    dataids = dataids[dataids.dataid.isin(filt.dataid)]
    dataids.pv.fillna(0, inplace=True)
    dataids.pv.replace({'yes': 1, 'no': 0}, inplace=True)

    dataids.total_amount_of_pv.fillna(0, inplace=True)
    dataids[dataids.total_amount_of_pv.eq(0)]['pv'] = 0

    filt['PV'] = filt['solar'].fillna(0) + filt['solar2'].fillna(0)
    filt['Load'] = filt['grid'] + filt['PV']
    filt.drop(columns=['solar', 'solar2'], inplace=True)
    filt.rename(columns={'grid': 'Net'}, inplace=True)

    allIDs_tx = dataids.dataid.drop_duplicates().sort_values().reset_index(drop=True)

    weather_tx = pd.read_csv('data/solcast_austin/csv_30.298683_-97.700371_fixed_23_180_PT15M.csv', usecols=[0,1,2,3])
    weather_tx['period_end'] = pd.to_datetime(weather_tx['period_end'], utc=True)
    weather_tx.set_index('period_end', inplace=True)
    weather_tx = weather_tx.tz_convert('US/Central')
    weather_tx.index = weather_tx.index.shift(periods=-1, freq='15T')
    weather_tx = weather_tx[weather_tx.index.year == 2018]

    latitude = 30.297
    longitude = -97.700
    location = Location(latitude=latitude, longitude=longitude)

    texas_dfs = []

    texas_df = pd.DataFrame(0, index=weather_tx.index, columns=['Net', 'PV', 'ghi', 'dhi', 'dni', 'PVkWp', 'location', 'Date', 'Time'])

    agg_tx = filt[filt['dataid'].isin(allIDs_tx)].groupby(filt.index).sum()
    agg_tx = agg_tx.reindex(texas_df.index).interpolate(method='time').fillna(0)

    texas_df['Net'] = agg_tx['Net'].values
    texas_df['PV'] = agg_tx['PV'].values
    texas_df['PVkWp'] = dataids[dataids['dataid'].isin(allIDs_tx)]['total_amount_of_pv'].sum()
    texas_df['ghi'] = weather_tx['ghi'].values
    texas_df['dni'] = weather_tx['dni'].values
    texas_df['dhi'] = weather_tx['dhi'].values
    texas_df['location'] = location

    texas_df['Date'] = texas_df.index.strftime('%Y-%m-%d')
    texas_df['Time'] = texas_df.index.strftime('%H:%M')


    texas_dfs.append(texas_df)


    texas_dfs, texas_subData, texas_Is, _ = extractFeatures_solar(texas_dfs, azimuths, tilts)

    tx_subDataDf = pd.DataFrame(texas_subData, columns=texas_subData[0].keys())
    tx_features = tx_subDataDf.drop(columns=['maxPV', 'baseline'])
    tx_features_scaled = scaler.transform(tx_features)

    return texas_dfs, tx_features_scaled, texas_Is, tx_subDataDf['maxPV']