from imports import *

nltk.download('vader_lexicon')
warnings.filterwarnings('ignore')

DATASET_1 = os.path.join("..", "Data", "stock_tweets.csv")
DATASET_2 =  os.path.join("..", "Data", "stock_yfinance_data.csv")