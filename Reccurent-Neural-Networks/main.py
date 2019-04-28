import os
import time

messege = "osascript -e " + "'" + "tell application " + '"' + "Messages" + '"' + " to send " + '"' + "I LOVE YOU <3" + '"' + " to buddy " + '"' + "Arianna Sterling" + '"' + " '"

print(messege)

#os.system(messege)


for i in range(10): 
	#time.sleep(0.1)
	os.system(messege)
