from abc import ABC, abstractmethod
import numpy as np
import csv


class BaseParser(ABC):
    def __init__(self):
        self._data_dict = {}

    @abstractmethod
    def get_data(self, ann_directory):
        pass


class ParserNoOne(BaseParser):
    """
        1   6   4  12   5   5   3   4   1  67   3   2   1   2   1   0   0   1   0   0   1   0   0   1   1
        2  48   2  60   1   3   2   2   1  22   3   1   1   1   1   0   0   1   0   0   1   0   0   1   2
        Last column is class, others are features.
    """

    # Load data from file
    def get_data(self, filename):
        # init the dataset as a list
        data = list()
        # open it as a readable file
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                row_line = line.rstrip().lstrip()
                row_arr = row_line.split()
                data.append(row_arr)

        data = np.array(data, np.int32)
        X = data[:, :-1]
        y = data[:, -1]
        return X, y


class ParserNoTwo(BaseParser):
    """
    battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram ,sc_h,sc_w,talk_time,three_g,touch_screen,wifi,price_range
    842          ,0   ,2.2        ,0       ,1 ,0     ,7         ,0.6  ,188      ,2      ,2 ,20       ,756     ,2549,9   ,7   ,19       ,0      ,0           ,1   ,1
    1021         ,1   ,0.5        ,1       ,0 ,1     ,53        ,0.7  ,136      ,3      ,6 ,905      ,1988    ,2631,17  ,3   ,7        ,1      ,1           ,0   ,2
    563          ,1   ,0.5        ,1       ,2 ,1     ,41        ,0.9  ,145      ,5      ,6 ,1263     ,1716    ,2603,11  ,2   ,9        ,1      ,1           ,0   ,2
    CSV File. Last column is class, others are features.
    """

    # Load data from file
    def get_data(self, filename):
        # init the dataset as a list
        data = list()
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    data.append(row)
                line_count += 1

        data = np.array(data, np.float)
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        return X, y
