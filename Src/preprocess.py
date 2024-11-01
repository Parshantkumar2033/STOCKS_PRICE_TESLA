from imports import *
import config

class DataPreparation:
    def __init__(self, dataset1, dataset2) -> None:
        '''
        dataset1 : ../Data/stock_tweets.csv
        dataset2 : ../Data/stock_yfinance_data.csv
        '''
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.stock_name = "TSLA"

    def technical_indicators(self, data):
        """
        Calculates and explains various moving averages and Bollinger Bands used in stock price analysis.

        1. **Moving Average (MA):**
            - **MA(7)**: Moving Average for the past 7 days, providing a short-term trend.
            - **MA(20)**: Moving Average for the past 20 days, offering a broader trend view.

        2. **Exponential Moving Average (EMA):**
            EMA gives more weight to recent prices, making it more responsive to new information. 
            The formula for EMA is:

            **EMA_t = P_close + (EMA_t-1 * (100 - P))**

            - *EMA_t*: EMA at time 't'
            - *P_close*: Closing price at time 't'
            - *EMA_t-1*: EMA of the previous time step
            - *P*: Smoothing factor (typically a percentage)

        3. **Bollinger Bands:**
            Bollinger Bands are volatility indicators consisting of three components:
            - **Middle Line**:  
                `middle line = stdev(MA(20))`
            - **Upper Bound**:  
                `upper bound = MA(20) + 2 * stdev(MA(20))`
            - **Lower Bound**:  
                `lower bound = MA(20) - 2 * stdev(MA(20))`
            These bands expand during high volatility and contract during low volatility, helping to identify potential overbought or oversold conditions.
        """
        data['MA7'] = data.iloc[:,4].rolling(window=7).mean() #Close column
        data['MA20'] = data.iloc[:,4].rolling(window=20).mean() #Close Column

        #This is the difference of Closing price and Opening Price
        data['MACD'] = data.iloc[:,4].ewm(span=26).mean() - data.iloc[:,1].ewm(span=12,adjust=False).mean()

        data['20SD'] = data.iloc[:, 4].rolling(20).std()
        data['upper_band'] = data['MA20'] + (data['20SD'] * 2)
        data['lower_band'] = data['MA20'] - (data['20SD'] * 2)

        data['EMA'] = data.iloc[:,4].ewm(com=0.5).mean()

        # Create LogMomentum
        data['logmomentum'] = np.log(data.iloc[:,4] - 1)

        return data

    def dataPreparation(self):
        all_tweets = pd.read_csv(self.dataset1)
        df = all_tweets[all_tweets['Stock Name'].isin([self.stock_name])]

        '''
        creating new empty columns ['sentiment_score', 'Negative', 'Neutral', 'Positive']
        Will store the sentiment_scores for each of the tweets
        '''
        temp_df = df.copy()
        temp_df["sentiment_score"] = ''
        temp_df["Negative"] = ''
        temp_df["Neutral"] = ''
        temp_df["Positive"] = ''


        '''
        CALCULATING THE SENTIMENT SCORE FOR EACH TWEET

        'NFKD' stands for Normalization Form KD (Compatibility Decomposition) It:
            1.Decomposes complex characters into simpler equivalents.
            2.Removes extra formatting or stylistic differences between visually 
              identical characters (like superscripts or accented letters).
        '''
        # 'from nltk.sentiment.vader import SentimentIntensityAnalyzer'
        sentiment_analyzer = SentimentIntensityAnalyzer()
        
        for indx, row in temp_df.T.iteritems():
            try:
                sentence_i = unicodedata.normalize('NFKD', temp_df.loc[indx, 'Tweet'])
                sentence_sentiment = sentiment_analyzer.polarity_scores(sentence_i)
                temp_df.at[indx, 'sentiment_score'] = sentence_sentiment['compound']
                temp_df.at[indx, 'Negative'] = sentence_sentiment['neg']
                temp_df.at[indx, 'Neutral'] = sentence_sentiment['neu']
                temp_df.at[indx, 'Positive'] = sentence_sentiment['pos']
            except TypeError:
                print (temp_df.loc[indx, 'Tweet'])
                print (indx)
                break

        temp_df['Date'] = pd.to_datetime(temp_df['Date'])   # date-time datatype correction
        temp_df['Date'] = temp_df['Date'].dt.date
        # dropping unecessary columns
        temp_df = temp_df.drop(columns=['Stock Name', 'Company Name', 'Negative', 'Positive', 'Neutral'])

        '''
        Grouping the data for each date by taking the mean('sentiment_score')
        '''
        tsla_df = temp_df.groupby([temp_df['Date']]).mean()

        # Fetching dataset2
        all_stocks = pd.read_csv(self.dataset2)

        stock_df = all_stocks[all_stocks['Stock Name'] == self.stock_name]
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df['Date'] = stock_df['Date'].dt.date
        final_df = stock_df.join(tsla_df, how="left", on="Date")
        final_df = final_df.drop(columns=['Stock Name'])


        # Adding techinal Indicators
        tech_df = self.technical_indicators(final_df)
        dataset = tech_df.iloc[20:,:].reset_index(drop=True)

        # visualizing the data
        plots.Utils.plot_price_time(final_df, 'price_vs_time.png')
        plots.Utils.plot_technical_indicators(tech_df, "technical_indicators.png")

        dataset.iloc[:, 1:] = pd.concat([dataset.iloc[:, 1:].ffill()])
        datetime_series = pd.to_datetime(dataset['Date'])
        datetime_index = pd.DatetimeIndex(datetime_series.values)
        dataset = dataset.set_index(datetime_index)
        dataset = dataset.sort_values(by='Date')
        dataset = dataset.drop(columns='Date')

        return dataset

class Preprocess:
    def __init__(self) -> None:
        pass

    def normalize_data(df: pd.DataFrame, 
                    range: Tuple[int, int],
                    target_column: str) -> Tuple[np.ndarray, np.ndarray]:        
        '''
        df: dataframe object
        range: type tuple -> (lower_bound, upper_bound)
            lower_bound: int
            upper_bound: int
        target_column: type str -> should reflect closing price of stock
        '''

        target_df_series = pd.DataFrame(df[target_column])
        data = pd.DataFrame(df.iloc[:, :]) 

        X_scaler = MinMaxScaler(feature_range=range)
        y_scaler = MinMaxScaler(feature_range=range)
        X_scaler.fit(data)
        y_scaler.fit(target_df_series)

        X_scale_dataset = X_scaler.fit_transform(data)
        y_scale_dataset = y_scaler.fit_transform(target_df_series)
        
        dump(X_scaler, open(config.X_SCALER_PKL, 'wb'))
        dump(y_scaler, open(config.Y_SCALER_PKL, 'wb'))
        return X_scale_dataset, y_scale_dataset
        
    def batch_data(x_data: Union[np.ndarray, List[List[float]]], 
               y_data: Union[np.ndarray, List[List[float]]], 
               batch_size: int, 
               predict_period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        X_batched, y_batched, yc = list(), list(), list()

        for i in range(0,len(x_data),1):
            x_value = x_data[i: i + batch_size][:, :]
            y_value = y_data[i + batch_size: i + batch_size + predict_period][:, 0]
            yc_value = y_data[i: i + batch_size][:, :]
            if len(x_value) == batch_size and len(y_value) == predict_period:
                X_batched.append(x_value)
                y_batched.append(y_value)
                yc.append(yc_value)

        return np.array(X_batched), np.array(y_batched), np.array(yc)

    def split_train_test(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:

        if len(data) < 20:
            raise ValueError("Input data must have at least 20 samples to split into train and test sets.")

        train_size = len(data) - 20
        data_train = data[0:train_size]
        data_test = data[train_size:]
        return data_train, data_test

    def predict_index(self, dataset: pd.DataFrame, 
                    X_train: pd.DataFrame, 
                    batch_size: int, 
                    prediction_period: int) -> Tuple[pd.Index, pd.Index]:
        """
        Retrieves the indices for training and testing predictions.

        Parameters:
        - dataset (pd.DataFrame): The complete dataset from which to derive indices.
        - X_train (pd.DataFrame): The training dataset used for predictions.
        - batch_size (int): The number of samples in each batch.
        - prediction_period (int): The number of time steps to predict.

        Returns:
        - Tuple[pd.Index, pd.Index]: A tuple containing the indices for training and testing predictions.
        """
        
        train_predict_index = dataset.iloc[batch_size: X_train.shape[0] + batch_size + prediction_period, :].index
        test_predict_index = dataset.iloc[X_train.shape[0] + batch_size:, :].index

        return train_predict_index, test_predict_index