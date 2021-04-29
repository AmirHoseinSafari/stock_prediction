from data_processing import load_data
from model import lstm_spy_zscores

if __name__ == '__main__':
    data = load_data.load_data_csv('Data/EOD_SPY_ZScores_Classified.csv')
    lstm_spy_zscores.run_model(data=data)