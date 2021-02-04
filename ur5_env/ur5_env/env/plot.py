import matplotlib.pyplot as plt

f = open("/run/media/morten/C01B-395D/plot.txt", "r")
plt_actions = []
plt_states = []
action = []

for line in f:
    if line[0] == "O":
        plt_states.append(line[3:])
    if line[0] == "A":
        plt_actions.append(line[3:])

print(plt_actions)

for i in range(len(plt_actions)):
    for j in range(len(plt_actions[i])):
        action[j].append(plt_actions[i][j])

plt.plot(action[0])
plt.plot(action[1])
plt.plot(action[2])
plt.plot(action[3])
plt.plot(action[4])
plt.plot(action[5])
plt.show()
