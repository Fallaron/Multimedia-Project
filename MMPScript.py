import os
import sys;
from scipy.optimize import curve_fit
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

    array1x = []
    array1y = []
    for line in file.readlines():
        if c == 0:
            if (int(line) == 0):
                arrayx = array0x
                arrayy = array0y
            if (int(line) == 1):
                arrayx = array1x
                arrayy = array1y
        if c == 1:
            treshold = float(line)
        if c == 2:
            truth = float(line)
        if c == 3:
            false_pos = float(line)
        if c == 4:
            num_gBoxes = float(line)

            falseRate = 1. - (false_pos / num_gBoxes)
            missedRate = 1. - (truth / num_gBoxes)

            arrayx.append(falseRate)
            arrayy.append(missedRate)
        c = (c + 1) % 5
    print array0y
    print array1y
    print array0x
    print array1x

    plt.ylabel('missed detection rate')
    plt.xlabel('false positive rate')
    plt.plot(array0x, array0y, 'b-', label='SVM x.0')

    plt.ylabel('missed detection rate')
    plt.xlabel('false positive rate')
    plt.plot(array1x, array1y, 'r-', label='SVM x.1')
    plt.legend(loc='upper right')
    plt.show()


main()
