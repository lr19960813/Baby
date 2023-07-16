import numpy as np
import matplotlib.pyplot as plt



class EarlyStopping():
    def __init__(self, tolerance = 5, min_delta = 0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True



def plot_confusion_matrix(cm, savename, title = 'Confusion Matrix'):
    plt.figure(figsize = (12, 8), dpi = 100)
    np.set_printoptions(precision = 2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color = 'red', fontsize = 15, va = 'center', ha = 'center')

    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation = 90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor = True)
    plt.gca().set_yticks(tick_marks, minor = True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which = 'minor', linestyle = '-')
    plt.gcf().subplots_adjust(bottom = 0.15)

    # show confusion matrix
    plt.savefig(savename, format = 'png')
    plt.show()
