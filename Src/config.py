from imports import *

nltk.download('vader_lexicon')
warnings.filterwarnings('ignore')

DATASET_1 = os.path.join("..", "Data", "stock_tweets.csv")
DATASET_2 =  os.path.join("..", "Data", "stock_yfinance_data.csv")

PLOT_PRICE_TIME = os.path.join("..", "Output", "Data_outputs") # path : ../Output/Data_outputs/
PLOT_TECHNICAL_INDICATORS = os.path.join("..", "Output", "Data_outputs")

TRAINING_PLOT = os.path.join("..", "Output", "Training")
TRAINING_RESULTS_PLOT = os.path.join("..", "Output", "Training")

TEST_PLOT = os.path.join("..", "Output", "Test")
TEST_RESULT_PLOT = os.path.join("..", "Output", "Test")

# normalized data
X_SCALED_PKL = os.path.join("..", "Data", "norm_data", "X_scaled.pkl")
Y_SCALED_PKL = os.path.join("..", "Data", "norm_data", "Y_scaled.pkl")


'''
Generator Model(versions) : ../Models/Generator/TSLA/
Discriminator Model(versions) : ../Models/Discriminator/TSLA/
'''