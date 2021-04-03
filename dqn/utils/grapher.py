from matplotlib import pyplot as plt

class Grapher(object):

    def __init__(self, title, x_axis, y_axis, leg_loc):

        self.title = title
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.leg_loc = leg_loc
        self.save_path = title + '/img{}.png'
        self.counter = 0

    def save(self):
        plt.savefig(self.save_path.format(self.counter))

    def update(self, data, recording=False):

        """
        param: data - [["Name", Array], ["Name", Array]]
        """
        self.counter += 1
        plt.title(self.title)
        plt.xlabel(self.x_axis)
        plt.ylabel(self.y_axis)
        for elmnt in data:
            plt.plot(elmn[1], label=elmnt[0])
        plt.legend(loc=self.leg_loc)
        plt.draw()
        if recording: self.save()
        plt.pause(0.0001)
        plt.clf()
