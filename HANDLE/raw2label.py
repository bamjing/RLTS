import pandas as pd
import numpy as np
import fastcluster
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering, Birch
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from Module import Config

class Handle_Data(object):
    def __init__(self, para_dict):
        RAW_PATH = para_dict['RAW_PATH']
        COLUMNS_TITLE = para_dict['COLUMNS_TITLE']

        self.LABELED_PATH = para_dict['LABELED_PATH']
        self.RAW_NAME = para_dict['RAW_NAME']
        self.CLUSTERING_LIST = para_dict['CLUSTERING_LIST']
        self.NUM_CLUSTERING = para_dict['NUM_CLUSTERING']
        self.BIRCH_threshold = para_dict['BIRCH_threshold']

        dateparse = lambda dates: [pd.datetime.strptime(d, '%Y%m%d  %H:%M:%S') for d in dates]
        csv_path = RAW_PATH + self.RAW_NAME + '.csv'
        self.df = pd.read_csv(csv_path, parse_dates=True, date_parser=dateparse, header=None)
        self.df.columns = COLUMNS_TITLE
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.clustering_algo = self.CLUSTERING_LIST[2]

    def _add_Label(self, urbl_data, labels):
        label_column = pd.DataFrame({'LABEL': labels})
        return urbl_data.merge(label_column, left_index=True, right_index=True)

    def processing_data(self):
        data = self._add_URBL()
        print "added urbl data."
        print "clustering..."
        urbl = np.array(data.as_matrix(), dtype=pd.Series)[:, 7:10]
        labels = self._clustering(self.clustering_algo, self.NUM_CLUSTERING, urbl)
        print "clustering finished."

        print "adding labeled data"
        labeledCSV = self._add_Label(data, labels)

        print "adding converted percentage volume data..."
        percentage_volume_addedCSV = self._add_volume_percent(labeledCSV)

        path = self.LABELED_PATH + self.RAW_NAME + '-' + self.clustering_algo + '-' + str(self.NUM_CLUSTERING) + '-labeled.csv'
        percentage_volume_addedCSV.to_csv(path)
        print "saved labeled and percentage volume file successfully!"
        self._visulization_clustering(self.clustering_algo, labels, urbl)

    def _add_volume_percent(self, labeledCSV):
        day_begin_list = self.df[(self.df['Date'].dt.hour == 13) & (self.df['Date'].dt.minute == 55)].index.tolist()
        if day_begin_list[0] != 0:
            day_begin_list[0:0] = [0]
        if day_begin_list[-1] != len(self.df) - 1:
            day_begin_list.append(len(self.df) - 1)

        volume_percent_list = []
        day_index = 0
        one_sum = 1
        for index in range(len(self.df)):

            if index == day_begin_list[day_index] and day_index + 1 < len(day_begin_list):
                one_day = self.df.iloc[day_begin_list[day_index]: day_begin_list[day_index + 1]]
                one_sum = one_day['Volume'].values.sum()
                day_index += 1
            percent_volume = (float(self.df['Volume'][index]) / float(one_sum)) * 100
            volume_percent_list.append(percent_volume)

        volume_percent_column = pd.DataFrame({'VOLUME_P': volume_percent_list})
        return labeledCSV.merge(volume_percent_column, left_index=True, right_index=True)

    def _add_URBL(self):
        convertedCSV = pd.DataFrame()
        for index in range(len(self.df)):
            Date = self.df['Date'][index]

            Open = self.df['Open'][index]
            Close = self.df['Close'][index]
            High = self.df['High'][index]
            Low = self.df['Low'][index]

            Count = self.df['Count'][index]
            Volume = self.df['Volume'][index]

            upperEdge = max(Open, Close)
            lowerEdge = min(Open, Close)

            upperShadow = abs(High - upperEdge)
            lowerShadow = abs(lowerEdge - Low)
            realBody = Close - Open

            column = pd.DataFrame({'Date': Date, 'Open': Open, 'Close': Close, 'High': High, 'Low': Low, 'Count': Count, 'Volume': Volume,
                                      'US': upperShadow, 'RB': realBody, 'LS': lowerShadow},
                                     index=[index],
                                     columns=['Date', 'Open', 'Close', 'High', 'Low', 'Count', 'Volume',
                                              'US', 'RB', 'LS'])
            convertedCSV = pd.concat([convertedCSV, column])

            if (index % 1000 == 0):
                print 'adding URBL data... %.2f' % (float(len(convertedCSV) * 100 / float(len(self.df)))), '% finished.'
        return convertedCSV

    def _clustering(self, name, num_labels, urbl_data):
        labels = []
        if name == 'fast_clustering':
            linkages = fastcluster.linkage_vector(urbl_data, method='ward')
            labels = fcluster(linkages, num_labels, criterion="maxclust")
        elif name == 'hierarchicalclustering':
            ward = AgglomerativeClustering(n_clusters=num_labels, linkage='ward').fit(urbl_data)
            labels = ward.labels_
        elif name == 'birch_clustering':
            model = Birch(threshold=self.BIRCH_threshold, n_clusters=num_labels, branching_factor=len(self.df))
            labels = model.fit_predict(urbl_data)
        return labels

    def _visulization_clustering(self, name, labels, urbl_data):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.view_init(7, -80)
        for l in np.unique(labels):
            ax.plot3D(urbl_data[labels == l, 0], urbl_data[labels == l, 1], urbl_data[labels == l, 2], 'o', color=plt.cm.jet(np.float(l) / np.max(labels + 1)))
        plt.title(name)
        plt.show()

if __name__ == "__main__":
    config = Config.Configuration()
    para_dict = config.parameter_dict
    Handle_Data = Handle_Data(para_dict)
    Handle_Data.processing_data()


