from temporalCNN.data_preprocessing import data_together, train_test_dataset
import talos as ta
# from temporalCNN.model_support import hyper_parameters, write_config, save_model_history
# from temporalCNN.neural_network import ssim_model
from talos.metrics.keras_metrics import fmeasure_acc
from talos import Deploy
import os
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping
from keras.layers import Dense,Flatten
from keras.models import Input, Model
from keras.utils import multi_gpu_model

from tcn import TCN


def optimizer_talos(datapath, final=False):
    # Default Parameters
    sampling_params = {
        'dim_in': 6,
        'output_length': 48,
        'min_before': 192,
        'max_before': 192,
        'min_after': 0,
        'max_after': 0,
        'model_number': 22
    }

    alldata, allpaths = data_together(datapath)


    train_x, train_y_do, train_y_temp, test_x, test_y_do, test_y_temp,scaler_x, scaler_do, scaler_temp = train_test_dataset(alldata[0], sampling_params)
    # Prepare model

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


        model_save_path = '/OSM/CBR/AF_WQ/source/Franz/Keras/TCNN_Model'

        outputdir_each = os.path.abspath(os.path.join(model_save_path, str(sampling_params['model_number'])))
        if not os.path.exists(outputdir_each):
            os.makedirs(outputdir_each)

        model_file = os.path.join(outputdir_each,
                                  '-{val_loss:.4f}-{epoch:02d}' + '.hdf5')
        checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_loss', save_best_only=False,
                                     save_weights_only=False, mode='auto', period=1)

        model_csv = os.path.join(outputdir_each,
                                 'train-history' + '.csv')
        csv_logger = CSVLogger(model_csv, append=True, separator=';')

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

        checkpoint_list = [checkpoint, csv_logger]






        # # Train the final model
        #
        # model_file = os.path.join('./output/tcnn_multi/',
        #                           '-{val_loss:04f}-{epoch:02d}' + '.hdf5')
        # checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_loss', save_best_only=False,
        #                              save_weights_only=False, mode='auto', period=1)
        #
        # model_csv = os.path.join('./output/tcnn_multi/',
        #                          'train-history' + '.csv')
        # csv_logger = CSVLogger(model_csv, append=True, separator=';')
        #
        # checkpoint_list = [checkpoint, csv_logger, early_stopping_callback]

        ## Model

        i = Input(batch_shape=(None,sampling_params['min_before'], sampling_params['dim_in']))

        o = TCN(kernel_size=3,dilations=[1,2,4,8,16,32,64],dropout_rate=0.6,return_sequences=True)(i)  # The TCN layers are here.
        o1 = Dense(64, activation='relu')(o)
        o2 = Dense(64, activation='relu')(o)
        o1 = Flatten()(o1)
        o2 = Flatten()(o2)
        out_1 = Dense(48)(o1)
        out_2 = Dense(48)(o2)

        m = Model(inputs=[i], outputs=[out_1, out_2])
        #m = Model(inputs=[i], outputs=[out_1])


        parallel_model = multi_gpu_model(m, gpus=2)
        parallel_model.compile(optimizer='adam', loss=['mse', 'mse'])
        history  = parallel_model.fit(train_x, [train_y_do, train_y_temp], epochs=10000, validation_split=0.1, callbacks=checkpoint_list)


        #parallel_model = multi_gpu_model(m, gpus=2)
        #parallel_model.compile(optimizer='adam', loss='mse')
        #history  = parallel_model.fit(train_x, train_y_temp, epochs=10000, validation_split=0.1, callbacks=checkpoint_list)


        #parallel_model = multi_gpu_model(m, gpus=2)
        #m.compile(optimizer='adam', loss=['mse', 'mse'])
        #history = m.fit(train_x, [train_y_do, train_y_temp], epochs=10000, validation_split=0.2, callbacks=checkpoint_list)

        # history = model.fit(x_train, y_train,
        #                     epochs=model_params['epochs'],
        #                     batch_size=model_params['batch_size'], shuffle=True,
        #                     # validation_split=model_params['validation_split'],
        #                     validation_data=(x_test, y_test),
        #                     callbacks=checkpoint)


        # # write hyperparameters for the model
        # write_config(final_hyper_parameters_regressor, './output/tcnn_multi/')

        # model.compile(loss=model_params['loss'],
        #               optimizer=model_params['optimizer'](
        #                   lr=lr_normalizer(model_params['lr'], model_params['optimizer'])),
        #               metrics=['acc', 'mse', 'mae', rsquare])
        #
        # model.summary()
        #
        # if not (checkpoint is None):
        #     # add callbacks
        #     history = model.fit(x_train, y_train,
        #                         epochs=model_params['epochs'],
        #                         batch_size=model_params['batch_size'], shuffle=True,
        #                         # validation_split=model_params['validation_split'],
        #                         validation_data=(x_test, y_test),
        #                         callbacks=checkpoint)









        # history, ssim = ssim_model(train_x, train_y, test_x, test_y, final_hyper_parameters,
        #                            checkpoint_list)
        #
        # write_config(final_hyper_parameters, outputdir_each)
        # save_model_history(history, outputdir_each)

        print('finish_final_model!')



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
    # datapath = r'C:\Users\ZHA244\Coding\QLD\burnett_river\fortcnn'
    datapath = '/OSM/CBR/AF_WQ/source/Franz/Keras_tcnn/multi_target/test_data/'
    optimizer_talos(datapath, True)
