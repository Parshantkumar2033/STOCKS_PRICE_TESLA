from imports import *
import config

class Utils:
    def __init__(self):
        self.stock_name = "TSLA"
    
    def plot_price_time(self, data: Union[pd.DataFrame, List[List[Any]], np.ndarray], output_file: str):
        '''
        Plots the stock price data and saves it as a PNG file.

        Parameters:
        - data (Union[List[List[Any]], np.ndarray]): A 2D array-like structure containing stock price data
        - output_file (str): The file path to save the plot, must end with '.png'.
        Saved at ../Output/Data_outputs/

        Raises:
        - ValueError: If the output file does not end with '.png'.
        '''
        print("Plotting : Price vs Time")

        if not output_file.endswith('.png'):
            raise ValueError("The output file must have a '.png' extension.")
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(data['Date'], data['Close'], color='#008B8B')
        ax.set(xlabel="Date", ylabel="USD", title=f"Stock Price")
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))

        output_directory = os.path.join(config.PLOT_PRICE_TIME) 
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        output_path = os.path.join(config.PLOT_PRICE_TIME, output_file)
        plt.savefig(output_path, bbox_inches='tight')

    def plot_technical_indicators(self, dataset, output_file: str) -> None:
        """
        Plots technical indicators including moving averages and closing price, and saves the plot as a PNG file.

        Parameters:
        - dataset (Union[List[List[Any]], np.ndarray]): A 2D array-like structure containing stock price data, 
        which must include 'Date', 'MA7', 'Close', and 'MA20' columns.
        - output_file (str): The file path to save the plot, must end with '.png'.
        
        Raises:
        - ValueError: If the dataset does not contain required columns or if the output file is not a .png file.
        - TypeError: If the dataset is not a list or NumPy array.
        """
        print("Plotting : Technical Indicators")
        if not output_file.endswith('.png'):
            raise ValueError("The output file must have a '.png' extension.")

        if isinstance(dataset, pd.DataFrame):
            df = pd.DataFrame(dataset, columns=['Date', 'Close', 'MA7', 'MA20'])
        elif isinstance(dataset, list):
            df = pd.DataFrame(dataset, columns=['Date', 'Close', 'MA7', 'MA20'])
        else:
            raise TypeError("Dataset must be either a 2D list or a NumPy array.")

        required_columns = ['Date', 'Close', 'MA7', 'MA20']

        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain the following columns: {required_columns}")

        fig, ax = plt.subplots(figsize=(15, 8), dpi=200)
        
        ax.plot(df['Date'], df['MA7'], label='Moving Average (7 days)', color='g', linestyle='--')
        ax.plot(df['Date'], df['Close'], label='Closing Price', color='#6A5ACD')
        ax.plot(df['Date'], df['MA20'], label='Moving Average (20 days)', color='r', linestyle='-.')
        
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        plt.title('Technical Indicators')
        plt.ylabel('Close (USD)')
        plt.xlabel("Year")
        plt.legend()

        output_directory = os.path.join(config.PLOT_TECHNICAL_INDICATORS) 
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        plt.savefig(os.path.join(config.PLOT_TECHNICAL_INDICATORS, output_file), bbox_inches='tight')

    def training_plot(self, discriminator_loss: List[Any], generator_loss: List[Any], output_file: str):
        print("Plotting : Training data plot...")
        plt.subplot(2, 1, 1)
        plt.plot(discriminator_loss, label='Disc_loss', color='#000000')
        plt.xlabel('Epoch')
        plt.ylabel('Discriminator Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(generator_loss, label='Gen_loss', color='#000000')
        plt.xlabel('Epoch')
        plt.ylabel('Generator Loss')
        plt.legend()

        if not output_file.endswith('.png'):
            raise ValueError("plots/training_plot/ : The output file must have a '.png' extension.")

        output_directory = config.TRAINING_PLOT
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        plt.savefig(os.path.join(output_directory, output_file), bbox_inches='tight')
        plt.show()

    def plot_traning_results(self, Real_price, Predicted_price, index_train, output_file: str, output_dim):
        print("Plotting : Training results plot...")

        X_scaler = load(open(config.X_SCALED_PKL, 'rb'))
        y_scaler = load(open(config.Y_SCALED_PKL, 'rb'))
        train_predict_index = index_train

        rescaled_Real_price = y_scaler.inverse_transform(Real_price)
        rescaled_Predicted_price = y_scaler.inverse_transform(Predicted_price)

        predict_result = pd.DataFrame()
        for i in range(rescaled_Predicted_price.shape[0]):
            y_predict = pd.DataFrame(rescaled_Predicted_price[i], columns=["predicted_price"], index=train_predict_index[i:i + output_dim])
            predict_result = pd.concat([predict_result, y_predict], axis=1, sort=False)
    
        real_price = pd.DataFrame()
        for i in range(rescaled_Real_price.shape[0]):
            y_train = pd.DataFrame(rescaled_Real_price[i], columns=["real_price"], index=train_predict_index[i:i + output_dim])
            real_price = pd.concat([real_price, y_train], axis=1, sort=False)
    
        predict_result['predicted_mean'] = predict_result.mean(axis=1)
        real_price['real_mean'] = real_price.mean(axis=1)

        plt.figure(figsize=(16, 8))
        plt.plot(real_price["real_mean"])
        plt.plot(predict_result["predicted_mean"], color='r')
        plt.xlabel("Date")
        plt.ylabel("Stock price")
        plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
        plt.title("The result of Training", fontsize=20)

        if not output_file.endswith('.png'):
            raise ValueError("plots/plot_training_results/ : The output file must have a '.png' extension.")
 
        output_directory = config.TRAINING_RESULTS_PLOT
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        plt.savefig(os.path.join(output_directory, output_file), bbox_inches='tight')
        return predict_result

    def plot_test_data(self, real_price: Union[pd.Series, pd.DataFrame, List[Any], np.ndarray], 
                       predict_result: Union[pd.Series, pd.DataFrame, List[Any], np.ndarray], 
                       output_file: str) -> None:
        print("Plotting : Test data plot...")
        plt.figure(figsize=(16, 8))
        plt.plot(real_price["real_mean"], color='#00008B')
        plt.plot(predict_result["predicted_mean"], color='#8B0000', linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Stock price")
        plt.legend(("Real price", "Predicted price"), loc="upper left", fontsize=16)
        plt.title(f"Prediction on test data for {self.stock_name}", fontsize=20)

        if not output_file.endswith('.png'):
            raise ValueError("plots/plot_test_data/ : The output file must have a '.png' extension.")     

        output_directory = config.TEST_RESULT_PLOT
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        plt.savefig(os.path.join(output_directory, output_file), bbox_inches='tight')
