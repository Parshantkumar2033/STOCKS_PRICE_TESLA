from imports import *
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

def plot_results(Real_price, Predicted_price, index_train, output_dim):
    plot = plots.Utils()
    predict_result = plot.plot_traning_results(Real_price, Predicted_price, index_train, 'Training_results.png', output_dim)
    predicted = predict_result["predicted_mean"]
    real = real_price["real_mean"]
    For_MSE = pd.concat([predicted, real], axis = 1)
    RMSE = np.sqrt(mean_squared_error(predicted, real))
    print('-- Train RMSE -- ', RMSE)    

if __name__ == "__main__":
    stock_name = "TSLA"

    (X_train, X_test, y_train, y_test, yc_train, 
    yc_test, input_dim, feature_size, output_dim, 
    index_train, index_test)  = preprocess_data(config.DATASET_1, config.DATASET_2)

    # model_dispatcher
    learning_rate = 5e-4
    epochs = 500
    model = Model(learning_rate, epochs, X_train, output_dim)
    generator, discriminator, g_optimizer, d_optimizer = model.make_model()

    # models_viz
    # plot_model(generator, to_file = os.path.join(config.MODEL_VIZ, 'generator_keras_model.png'), show_shapes=True)
    # plot_model(discriminator, to_file= os.path.join(config.MODEL_VIZ, 'discriminator_keras_model.png'), show_shapes=True)

    # training_and_testing
    train = Train()
    predicted_price, real_price, RMSPE = train.training(X_train, y_train, yc_train, epochs, generator, discriminator, g_optimizer, d_optimizer)

    # plot_training_results
    plot_results(real_price, predicted_price, index_train, output_dim)

    # Testing
    test_generator = test_generator = tf.keras.models.load_model(f'../Models/Generator/TSLA/generator_V_{epochs-1}.h5')

    test_model = test.Test()
    predicted_price = test_model.eval_op(test_generator, X_test)
    # data_plot(self, Real_test_price, Predicted_test_price, index_test, output_dim
    test_rmse = test_model.data_plot(y_test, predicted_price, index_test, output_dim)

    print("--Test RMSE : ", test_rmse)