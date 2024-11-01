from imports import *
import config

class Test:
    def __init__(self):
        pass

    @tf.function
    def eval_op(self, generator, real_x):
        generated_data = generator(real_x, training = False)
        return generated_data
    
    def data_plot(self, Real_test_price, Predicted_test_price, index_test, output_dim) -> float:
        X_scaler = load(open(config.X_SCALED_PKL, 'rb'))
        y_scaler = load(open(config.Y_SCALED_PKL, 'rb'))
        test_predict_index = index_test

        rescaled_Real_price = y_scaler.inverse_transform(Real_test_price)
        rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_test_price)

        predict_result = pd.DataFrame()
        for i in range(rescaled_Predicted_price.shape[0]):
            y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=["predicted_price"], index=test_predict_index[i:i+output_dim])
            predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
    
        real_price = pd.DataFrame()
        for i in range(rescaled_Real_price.shape[0]):
            y_train = pd.DataFrame(rescaled_Real_price[i], columns=["real_price"], index=test_predict_index[i:i+output_dim])
            real_price = pd.concat([real_price, y_train], axis=1, sort=False)
    
        predict_result['predicted_mean'] = predict_result.mean(axis=1)
        real_price['real_mean'] = real_price.mean(axis=1)

        predicted = predict_result["predicted_mean"]
        real = real_price["real_mean"]
        For_MSE = pd.concat([predicted, real], axis = 1)
        RMSE = np.sqrt(mean_squared_error(predicted, real))

        # data plot
        plots.Utils.plot_test_data(real_price, predict_result, 'test_data_plot.png')

        return RMSE