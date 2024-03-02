import time

map = {
    0: "\\",
    1: "|",
    2: "/",
    3: "-",
}

for i in range(0, 100):
    print("{}/100 {}".format(i, map[i % 4]), end="\r")
    time.sleep(0.10)