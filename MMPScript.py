
import sys

import matplotlib.pyplot as plt
import numpy


def update_line(hl, new_data):
    hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
    hl.set_ydata(numpy.append(hl.get_ydata(), new_data))





def main():
    if len(sys.argv) != 2:
        print "1 Argument needed!"
        return
    file = open(sys.argv[1], "r")
    c = 0
    array0x = []
    array0y = []
    array0z = []

    array1x = []
    array1y = []
    array1z = []
    for line in file.readlines():
        if c == 0:
            if (int(line) == 0):
                arrayx = array0x
                arrayy = array0y
                arrayz = array0z
            if (int(line) == 1):
                arrayx = array1x
                arrayy = array1y
                arrayz = array1z
        if c == 1:
            treshold = float(line)
        if c == 2:
            truth = float(line)
        if c == 3:
            false_pos = float(line)
        if c == 4:
            num_gBoxes = float(line)

            falseRate = (false_pos / num_gBoxes)*100
            missedRate = (1. - (truth / num_gBoxes))*100

            arrayx.append(falseRate)
            arrayy.append(missedRate)
            arrayz.append(treshold)

        c = (c + 1) % 5
    print array0y
    print array1y
    print array0x
    print array1x
    fig, ax = plt.subplots();
    plt.ylabel('missed detection rate')
    plt.xlabel('false positive rate')
    ax.plot(array0x, array0y, '-bo', label='SVM x.0')

    for X,Y,Z in zip(array0x,array0y,array0z):
        ax.annotate('{}'.format(Z), xy=(X,Y), xytext=(-5, 5), ha='left',
                textcoords='offset points')


    ax.plot(array1x, array1y, '-ro', label='SVM x.1')

    for X,Y,Z in zip(array1x,array1y,array1z):
        ax.annotate('{}'.format(Z), xy=(X,Y), xytext=(-5, -25), ha='left',
                textcoords='offset points')
    plt.legend(loc='upper right')
    plt.show()


main()