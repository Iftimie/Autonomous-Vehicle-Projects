import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def slider():

    plt.subplots_adjust(left=0.25, bottom=0.25)
    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sfreq = Slider(axfreq, 'Speed', -100, 100, valinit=0)


    def update(val):
        #l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        print("dsfsdf")
    sfreq.on_changed(update)

    plt.show()

from threading import Thread
thread = Thread(target = slider)
thread.start()
import time
time.sleep(10)
