import os
import pickle
import matplotlib.pyplot as plt


def save_pickle(output_file, data):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'wb+') as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def plot_values(title, data, xlabel, ylabel, legend_location):
    plt.figure(title)
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for dlabel, dpoints in data:
        plt.plot(dpoints, label=dlabel)

    plt.legend(loc=legend_location)
    plt.pause(.001)


def show_plots():
    return plt.show()