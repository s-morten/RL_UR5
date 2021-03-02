import matplotlib.pyplot as plt

f = open("/run/media/morten/C01B-395D/plot.txt", "r")
plt_actions = []
plt_states = []
action0 = []
action1 = []
action2 = []
action3 = []
action4 = []
action5 = []
state0 = []
state1 = []
state2 = []
state3 = []
state4 = []
state5 = []

for line in f:
    if line[0] == "O":
        if line[1] == "0":
            state0.append(float(line[4:].rstrip()))
        if line[1] == "1":
            state1.append(float(line[4:].rstrip()))
        if line[1] == "2":
            state2.append(float(line[4:].rstrip()))
        if line[1] == "3":
            state3.append(float(line[4:].rstrip()))
        if line[1] == "4":
            state4.append(float(line[4:].rstrip()))
        if line[1] == "5":
            state5.append(float(line[4:].rstrip()))
    if line[0] == "A":
        if line[1] == "0":
            action0.append(float(line[4:].rstrip()))
        if line[1] == "1":
            action1.append(float(line[4:].rstrip()))
        if line[1] == "2":
            action2.append(float(line[4:].rstrip()))
        if line[1] == "3":
            action3.append(float(line[4:].rstrip()))
        if line[1] == "4":
            action4.append(float(line[4:].rstrip()))
        if line[1] == "5":
            action5.append(float(line[4:].rstrip()))


# axes = plt.gca()
# axes.set_xlim([0,200])
# axes.set_ylim([-3.1416,3.1416])
# plt.plot(action0)
# plt.plot(action1)
# plt.plot(action2)
# plt.plot(action3)
# plt.plot(action4)
# plt.plot(action5)
plt.plot(state0)
plt.plot(state1)
plt.plot(state2)
plt.plot(state3)
plt.plot(state4)
plt.plot(state5)
plt.legend({'action 0','action 1','action 2','action 3','action 4','action 5'})
plt.show()
