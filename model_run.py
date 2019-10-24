from temporalCNN.data_preprocessing import data_together, train_test_dataset
import talos as ta
# from temporalCNN.model_support import hyper_parameters, write_config, save_model_history
# from temporalCNN.neural_network import ssim_model
from talos.metrics.keras_metrics import fmeasure_acc
from talos import Deploy
import os
from keras.callbacks import ModelCheckpoint,CSVLogger, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, TensorBoard
from keras.layers import Dense, Reshape,Flatten, Activation
from keras.models import Input, Model, load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt, fabs


from sdae.DecoupleWeightDecay import AdamW, WeightDecayScheduler
from domain_regressor.model_support import lr_schedule

from tcn import TCN


# Seed value
# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)













def Kmean_output_shape(input_shape):
    return (input_shape[0], input_shape[2])

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    out = 0
    for i in range(y_true.shape[0]):
        a = y_true[i]
        b = y_pred[i]
        c = a+b
        if c == 0:
            continue
        out += fabs(a - b) / c
    out *= (200.0 / y_true.shape[0])
    return out


def optimizer_talos(datapath, final=False):
    # Default Parameters
    sampling_params = {
        'dim_in': 8,
        'output_length': 1,
        'min_before': 48,
        'max_before': 48,
        'min_after': 0,
        'max_after': 0,
        'model_number': 233
    }

    alldata, allpaths = data_together(datapath)


    train_x, train_y_do, train_y_temp, test_x, test_y_do, test_y_temp, scaler_x, scaler_do, scaler_temp = train_test_dataset(alldata[0], sampling_params)
    # Prepare model

    print(train_x[1])
    print(train_y_do[0])
    print(train_y_temp[0])

    # model_hyper_parameters,final_hyper_parameters = hyper_parameters()

    if not final:
        # Run the model
        h = ta.Scan(train_x, train_y, params=model_hyper_parameters,
                    model=ssim_model,
                    dataset_name='water_quality',
                    experiment_no='1',
                    random_method='quantum',
                    grid_downsample=.01,
                    reduction_method='correlation',
                    reduction_threshold=0.01,
                    reduction_interval=2,
                    reduction_metric=fmeasure_acc,
                    functional_model=True,
                    print_params=True)

        # Deploy the model

        Deploy(h, 'ssim_test')
        print('finish!')
    else:
        # Train the final model


        # model_save_path = '/OSM/CBR/AF_WQ/source/Franz/Keras/TCNN_Model'
        #
        # outputdir_each = os.path.abspath(os.path.join(model_save_path, str(sampling_params['model_number'])))
        # if not os.path.exists(outputdir_each):
        #     os.makedirs(outputdir_each)
        #
        # model_file = os.path.join(outputdir_each,
        #                           '-{val_loss:.4f}-{epoch:02d}' + '.hdf5')
        # checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_loss', save_best_only=False,
        #                              save_weights_only=False, mode='auto', period=1)
        #
        # model_csv = os.path.join(outputdir_each,
        #                          'train-history' + '.csv')
        # csv_logger = CSVLogger(model_csv, append=True, separator=';')
        #
        # checkpoint_list = [checkpoint, csv_logger]






        # Train the final model

        model_file = os.path.join('./temporalCNN/do_temp_newtest/results/',
                                  '-{val_loss:04f}-{epoch:02d}' + '.hdf5')
        checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_loss', save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)

        model_csv = os.path.join('./temporalCNN/do_temp_newtest/results/',
                                 'train-history' + '.csv')
        csv_logger = CSVLogger(model_csv, append=True, separator=';')


        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)

        wd_scheduler = WeightDecayScheduler(init_lr=lr_schedule(0))

        tb_cb = TensorBoard(log_dir='./temporalCNN/do_temp_newtest/results/log/')


        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

        checkpoint_list = [checkpoint, csv_logger, early_stopping_callback, lr_scheduler, lr_reducer, wd_scheduler, tb_cb]

        # Model

        i = Input(batch_shape=(None,sampling_params['min_before'], sampling_params['dim_in']))

        o = TCN(kernel_size=12, nb_stacks=1, dilations=[1,2,4,8,16],dropout_rate=0.05,return_sequences=True, use_skip_connections=True,)(i)
        o1 = Dense(8, activation='tanh')(o)
        o2 = Dense(8, activation='tanh')(o)
        o1 = Dense(8, activation='tanh')(o1)
        o2 = Dense(8, activation='tanh')(o2)
        # o1 = Dense(16, activation='relu')(o1)
        # o2 = Dense(16, activation='relu')(o2)
        o1 = Flatten()(o1)
        o2 = Flatten()(o2)
        out_1 = Dense(1)(o1)
        out_2 = Dense(1)(o2)
        out_1 = Activation('linear')(out_1)
        out_2 = Activation('linear')(out_2)
        # # # The TCN layers are here.
        # out_1 = Dense(1)(o)
        # out_2 = Dense(1)(o)

        m = Model(inputs=[i], outputs=[out_1, out_2])
        m.compile(optimizer=AdamW(weight_decay=1e-4, lr=lr_schedule(0)), loss=['mse', 'mse'])
        # m.compile(optimizer='adam', loss=['mse', 'mse'])
        m.summary()



        history  = m.fit(train_x, [train_y_do, train_y_temp], epochs=5000, validation_split=0.1, callbacks=checkpoint_list, batch_size=50)
        print('finish_final_model!')



        # #
        # #
        # trained_model = load_model('./output/tcnn_multi/dochlo/-0.0050-407.hdf5')
        # print(trained_model.summary())
        #
        #
        #
        # pred_do, pred_temp = trained_model.predict(test_x)
        #
        # print(pred_do.shape)
        # print(pred_temp.shape)
        #
        #
        #
        # pred_do = scaler_do.inverse_transform(pred_do)
        # pred_temp = scaler_temp.inverse_transform(pred_temp)
        # ori_do = scaler_do.inverse_transform(test_y_do)
        # ori_temp = scaler_temp.inverse_transform(test_y_temp)
        #
        #
        # # # write to a csv
        # #
        # # df = pd.DataFrame(pred_do)
        # # df2 = pd.DataFrame(pred_temp)
        # # df3 = pd.DataFrame(ori_do)
        # # df4 = pd.DataFrame(ori_temp)
        # #
        # # df.to_csv('./output/tcnn_multi/predictiondo.csv')
        # # df2.to_csv('./output/tcnn_multi/predictiontemp.csv')
        # # df3.to_csv('./output/tcnn_multi/orido.csv')
        # # df4.to_csv('./output/tcnn_multi/oritemp.csv')
        #
        #
        # print(pred_do.shape)
        # print(pred_temp.shape)
        # # print(ori_x.shape)
        #
        # print('Predictive Performance (all together):')
        #
        # mse_all = mean_squared_error(pred_do, ori_do)
        # print('RMSE (sklearn)_do: {0:f}'.format(sqrt(mse_all)))
        # mse_scalar = mean_squared_error(pred_temp, ori_temp)
        # print('RMSE (sklearn)_temp: {0:f}'.format(sqrt(mse_scalar)))
        # mae_all = mean_absolute_error(pred_do, ori_do)
        # print("MAE (sklearn)_do:{0:f}".format(mae_all))
        # mae_scalar = mean_absolute_error(pred_temp, ori_temp)
        # print("MAE (sklearn)_temp:{0:f}".format(mae_scalar))
        # r2_all = r2_score(pred_do, ori_do)
        # print("R2 (sklearn)_do:{0:f}".format(r2_all))
        # r2_scalar = r2_score(pred_temp, ori_temp)
        # print("R2 (sklearn)_temp:{0:f}".format(r2_scalar))
        # print("---------")
        # mape = mean_absolute_percentage_error(ori_do, pred_do)
        # print("MAPE (sklearn)_do:{0:f}".format(mape))
        # mape = mean_absolute_percentage_error(ori_temp, pred_temp)
        # print("MAPE (sklearn)_temp:{0:f}".format(mape))
        #
        #
        # # for i in range(50):
        # #     ## one case
        # #     ori_x = scaler_x.inverse_transform(test_x[i])
        # #     ori_do_all = np.concatenate([ori_x[:, 3], ori_do[i]])
        # #     zeroarray = np.full_like(ori_x[:, 3], np.nan)
        # #     pre_do = np.concatenate([zeroarray, pred_do[i]])
        # #
        # #     ori_temp_all = np.concatenate([ori_x[:, 0], ori_temp[i]])
        # #     zeroarray = np.full_like(ori_x[:, 0], np.nan)
        # #     pre_temp = np.concatenate([zeroarray, pred_temp[i]])
        # #
        # #     f, (ax1, ax2) = plt.subplots(1, 2)
        # #     ax1.plot(ori_do_all, 'b', marker='*', label='True')
        # #     ax1.plot(pre_do, 'r', marker='o', label='Predict')
        # #     ax2.plot(ori_temp_all, 'b', marker='*', label='True')
        # #     ax2.plot(pre_temp, 'r', marker='o', label='Predict')
        # #     plt.title(i)
        # #     plt.show()
        #
        #
        #
        # #
        # # ## one case
        # # ori_x = scaler_x.inverse_transform(test_x[26])
        # # ori_do_all = np.concatenate([ori_x[:,3], ori_do[26]])
        # # zeroarray = np.full_like(ori_x[:,3], np.nan)
        # # pre_do = np.concatenate([zeroarray, pred_do[26]])
        # #
        # # ori_temp_all = np.concatenate([ori_x[:,0], ori_temp[26]])
        # # zeroarray = np.full_like(ori_x[:,0], np.nan)
        # # pre_temp = np.concatenate([zeroarray, pred_temp[26]])
        # #
        # #
        # # f, (ax1, ax2) = plt.subplots(1, 2)
        # # ax1.plot(ori_do_all, 'b', marker='*', label='True')
        # # ax1.plot(pre_do, 'r', marker='o', label='Predict')
        # # ax2.plot(ori_temp_all, 'b', marker='*', label='True')
        # # ax2.plot(pre_temp, 'r', marker='o', label='Predict')
        # # plt.show()
        #
        #
        #
        #
        # # # single output
        # #
        # # trained_model = load_model('./output/tcnn_multi/single/-0.0011-874.hdf5')
        # # print(trained_model.summary())
        # #
        # # pred_do = trained_model.predict(test_x)
        # #
        # # print(pred_do.shape)
        # #
        # #
        # # # scaler_x, scaler_do, scaler_temp
        # # # scaler_x.fit(df[['Temp_degC', 'EC_uScm', 'pH', 'DO_mg', 'Turbidity_NTU', 'Chloraphylla_ugL']])
        # #
        # # pred_do = scaler_do.inverse_transform(pred_do)
        # # # pred_temp = scaler_temp.inverse_transform(pred_temp)
        # # ori_do = scaler_do.inverse_transform(test_y_do)
        # # # ori_temp = scaler_temp.inverse_transform(test_y_temp)
        # #
        # #
        # # print('Predictive Performance (all together):')
        # #
        # # mse_all = mean_squared_error(pred_do, ori_do)
        # # print('RMSE (sklearn)_do: {0:f}'.format(sqrt(mse_all)))
        # # # mse_scalar = mean_squared_error(pred_temp, ori_temp)
        # # # print('RMSE (sklearn)_temp: {0:f}'.format(sqrt(mse_scalar)))
        # # mae_all = mean_absolute_error(pred_do, ori_do)
        # # print("MAE (sklearn)_do:{0:f}".format(mae_all))
        # # # mae_scalar = mean_absolute_error(pred_temp, ori_temp)
        # # # print("MAE (sklearn)_temp:{0:f}".format(mae_scalar))
        # # r2_all = r2_score(pred_do, ori_do)
        # # print("R2 (sklearn)_do:{0:f}".format(r2_all))
        # # # r2_scalar = r2_score(pred_temp, ori_temp)
        # # # print("R2 (sklearn)_temp:{0:f}".format(r2_scalar))
        # # mape = mean_absolute_percentage_error(ori_do, pred_do)
        # # print("MAPE (sklearn)_do:{0:f}".format(mape))
        # # # mape = mean_absolute_percentage_error(ori_temp, pred_temp)
        # # # print("MAPE (sklearn)_temp:{0:f}".format(mape))
        #






def optimizer_sherpa(datapath):
    # Default Parameters
    sampling_params = {
        'dim_in': 11,
        'output_length': 5,
        'min_before': 2,
        'max_before': 7,
        'min_after': 2,
        'max_after': 7
    }

    alldata, allpaths = data_together(datapath)
    train_x, train_y, test_x, test_y = train_test_dataset(alldata[0], sampling_params)
    # Prepare model

    model_hyper_parameters = hyper_parameters()

    # Run the model


if __name__ == "__main__":
    # Prepare data
    datapath = r'C:\Users\ZHA244\Coding\QLD\burnett_river\fortcnn_resample'
    # datapath = '/OSM/CBR/AF_WQ/source/Franz/Keras_tcnn/multi_target/test_data/'
    optimizer_talos(datapath, True)
