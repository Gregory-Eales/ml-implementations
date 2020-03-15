from matplotlib import pyplot as plt

from utils.data_generator import generate_data
from rnn.rnn import RNN



x, y, sin_data = generate_data()


rnn = RNN(alpha=0.005)
rnn.optimize(x, y, iter=1000)

prediction = rnn.forward(x)

#plt.plot(sin_data.numpy())
print(prediction)
plt.plot(prediction.detach().numpy())
plt.show()
