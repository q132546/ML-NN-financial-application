import pandas as pd
import numpy as np
import json


class RealTime(object):
    def __init__(self):
        self.now_factors_data = pd.read_csv('sample_data/sample_factors_data18Q4.csv')
        self.config = pd.read_json('config.json')
        self.stocks_symbol = self.config['Finance']['stocks_pool']
        self.long_symbol = self.config['Finance']['long_symbol']
        self.short_symbol = self.config['Finance']['short_symbol']



if __name__ == '__main__':
    realtime = RealTime()
    print(realtime.now_factors_data)
