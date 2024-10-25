from imports import *

class Utils:
    def __init__(self):
        pass
    
    def plot_price_time(self, data : Union[List[List[Any]], np.ndarray], output_file : str):
        '''
        Plots the stock price data and saves it as a PNG file.

        Parameters:
        - data (Union[List[List[Any]], np.ndarray]): A 2D array-like structure containing stock price data
        - output_file (str): The file path to save the plot, must end with '.png'.
        Saved at ../Output/Data_outputs/

        Raises:
        - ValueError: If the output file does not end with '.png'.
        '''

        if not output_file.endswith('.png'):
            raise ValueError("The output file must have a '.png' extension.")
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(data['Date'], data['Close'], color='#008B8B')
        ax.set(xlabel="Date", ylabel="USD", title=f"Stock Price")
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))

        # Save the plot before showing it
        output_path = os.path.join("..", "Output", "Data_outputs", output_file)
        plt.savefig(output_path, bbox_inches='tight')

    def plot_technical_indicators(self, dataset: Union[List[List[Any]], np.ndarray], output_file: str) -> None:
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
        if not output_file.endswith('.png'):
            raise ValueError("The output file must have a '.png' extension.")

        if isinstance(dataset, np.ndarray):
            df = pd.DataFrame(dataset, columns=['Date', 'Close', 'MA7', 'MA20'])
        elif isinstance(dataset, list):
            df = pd.DataFrame(dataset, columns=['Date', 'Close', 'MA7', 'MA20'])
        else:
            raise TypeError("Dataset must be either a 2D list or a NumPy array.")

        required_columns = ['Date', 'Close', 'MA7', 'MA20']

        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain the following columns: {required_columns}")

        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 8), dpi=200)
        
        ax.plot(df['Date'], df['MA7'], label='Moving Average (7 days)', color='g', linestyle='--')
        ax.plot(df['Date'], df['Close'], label='Closing Price', color='#6A5ACD')
        ax.plot(df['Date'], df['MA20'], label='Moving Average (20 days)', color='r', linestyle='-.')
        
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        plt.title('Technical Indicators')
        plt.ylabel('Close (USD)')
        plt.xlabel("Year")
        plt.legend()

        plt.savefig(output_file, bbox_inches='tight')