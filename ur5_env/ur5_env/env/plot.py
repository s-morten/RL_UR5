import matplotlib.pyplot as plt

f = open("./../results.txt", "r")
plt_values = []
tmp = 0
cnt = 0

for line in f:
    rew = line.rstrip()
    # if float(rew) > -10:
    #     tmp += float(rew)
    #     cnt += 1
    tmp += float(rew)
    cnt += 1
    if cnt >= 100:
        cnt = 0
        plt_values.append(tmp / 100)
        tmp = 0

# plt_values.append(tmp)
plt.plot(plt_values)
plt.show()