from imports import *
import pickle 
import config
from preprocess import DataPreparation, Preprocess
from train import Train
from model_dispatcher import Model
import plots
import test

def preprocess_data(dataset1, dataset2):
    dprep = DataPreparation(dataset1, dataset2)
    preprocess = Preprocess()
    dataset = dprep.dataPreparation()

    X_scale_dataset,y_scale_dataset = preprocess.normalize_data(dataset, (-1,1), "Close")
    X_batched, y_batched, yc = preprocess.batch_data(X_scale_dataset, y_scale_dataset, batch_size = 5, predict_period = 1)
    X_train, X_test, = preprocess.split_train_test(X_batched)
    y_train, y_test, = preprocess.split_train_test(y_batched)
    yc_train, yc_test, = preprocess.split_train_test(yc)
    index_train, index_test, = preprocess.predict_index(dataset, X_train, 5, 1)
    input_dim = X_train.shape[1] 
    feature_size = X_train.shape[2] 
    output_dim = y_train.shape[1]
    return X_train, X_test, y_train, y_test, yc_train, yc_test, input_dim, feature_size, output_dim, index_train, index_test

def find_rmse(predict_result, real_price):
    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]
    For_MSE = pd.concat([predicted, real], axis = 1)
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('-- Train RMSE -- ', RMSE)
    return RMSE

def pickle_file(data):
    # file_save = os.path.join("..", "Models", "Important_variables.pkl")
    # if not os.path.exists(file_save):
    #     os.makedirs(file_save)
    with open('Important_variables.pkl', "wb") as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    stock_name = "TSLA"

    (X_train, X_test, y_train, y_test, yc_train, 
    yc_test, input_dim, feature_size, output_dim, 
    index_train, index_test)  = preprocess_data(config.DATASET_1, config.DATASET_2)
    data_prepared = (X_train, X_test, y_train, y_test, yc_train, 
                    yc_test, input_dim, feature_size, output_dim, 
                    index_train, index_test)
    pickle_file(data_prepared)

    # model_dispatcher
    learning_rate = 5e-4
    epochs = 500
    model = Model(learning_rate, epochs, X_train, output_dim)
    generator, discriminator, g_optimizer, d_optimizer = model.make_model()

    model_outputs = (generator, discriminator, g_optimizer, d_optimizer)
    pickle_file(model_outputs)


    # models_viz
    # plot_model(generator, to_file = os.path.join(config.MODEL_VIZ, 'generator_keras_model.png'), show_shapes=True)
    # plot_model(discriminator, to_file= os.path.join(config.MODEL_VIZ, 'discriminator_keras_model.png'), show_shapes=True)

    # training_and_testing
    train = Train()
    predicted_price, real_price, RMSPE = train.training(X_train, y_train, yc_train, epochs, generator, discriminator, g_optimizer, d_optimizer)

    training_result = (predicted_price, real_price, RMSPE)
    pickle_file(training_result)


    # plot_training_results
    plot = plots.Utils()
    predict_result, real_price = plot.plot_traning_results(real_price, predicted_price, index_train, 'Training_results.png', output_dim)
    training_rmse = find_rmse(predict_result, real_price)
    plotted = (predict_result, real_price, training_rmse)
    pickle_file(plotted)


    # Testing
    test_generator = test_generator = tf.keras.models.load_model(os.path.join("..", "Models", "Generator", "TSLA", "generator_V_499.h5"))

    test_model = test.Test()
    predicted_price = test_model.eval_op(test_generator, X_test)
    # data_plot(self, Real_test_price, Predicted_test_price, index_test, output_dim

    pickle_file(predicted_price)

    test_real_price, test_rmse = test_model.data_plot(y_test, predicted_price, index_test, output_dim)

    pickle_file(test_rmse)
    print("--Test RMSE--", test_rmse)