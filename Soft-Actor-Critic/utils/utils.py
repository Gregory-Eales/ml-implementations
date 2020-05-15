from matplotlib import pyplot as plt
import numpy as np

def graph(sac, avg_reward, disc_r):
	

	plt.title("Reward per Epoch")
	plt.xlabel("Epoch")
	plt.ylabel("Reward")

			
	ls1 = np.linspace(0, len(sac.q1_loss), num=len(avg_reward)).tolist()
	avg_rp = np.array(avg_reward)
	plt.plot(ls1, avg_rp/np.max(np.abs(avg_rp)), label="Reward")

	q1_loss = np.array(sac.q1_loss)
	q1_loss = q1_loss/q1_loss.max()

	q2_loss = np.array(sac.q2_loss)
	q2_loss = q2_loss/q2_loss.max()
	
	p_loss = np.array(sac.p_loss)
	p_loss = p_loss/abs(np.max(np.abs(p_loss)))

	plt.plot(q1_loss, label="Q1 loss")
	plt.plot(q2_loss, label="Q2 loss")

	ls2 = np.linspace(0, len(sac.q1_loss), num=len(p_loss)).tolist()
	plt.plot(ls2, p_loss, label="P loss")

	disc_rp = np.array(disc_r)
	disc_rp = disc_rp/np.max(np.abs(disc_rp))
	ls3 = np.linspace(0, len(sac.q1_loss), num=len(disc_rp)).tolist()
	
	plt.plot(ls3, disc_rp, label="Discount Reward")

	plt.legend()
	plt.draw()
	plt.pause(0.0001)
	plt.clf()