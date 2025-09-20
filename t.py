from motors_i2c import Motor
import matplotlib.pyplot as plt
import numpy as np
from time import sleep, perf_counter as now

MOTOR_TOPRIGHT = 0x19
MOTOR_TOPLEFT = 0x1a
MOTOR_BOTTOMRIGHT = 0x1c
MOTOR_BOTTOMLEFT = 0x1b
DRIBBLER = 0x1e
motors = {
    0: Motor(address=MOTOR_TOPRIGHT),
    1: Motor(address=MOTOR_TOPLEFT),
    3: Motor(address=MOTOR_BOTTOMRIGHT),
    2: Motor(address=MOTOR_BOTTOMLEFT),
    "dribbler": Motor(address=DRIBBLER, max_speed=167_000_000)
}

def plot_current():
    [motors[j].set_speed(0.5) for j in range(4)]

    N = 200

    t = np.zeros(N)
    y_lists = [np.zeros(N, dtype=np.float64) for _ in range(4)]

    initial_t = now()
    for i in range(N):
        t[i] = now() - initial_t
        for j in range(4):
            data = motors[j].read()
            y_lists[j][i] = data[1]
        sleep(0.0001)

    [motors[j].set_speed(0) for j in range(4)]

    for j in range(4):
        plt.plot(t, y_lists[j])

    plt.xlabel("time")
    plt.ylabel("current")
    plt.title("bingus")

    plt.show()

def plot_deltaCurrent():
    try:
        threshold = 0.9829
        
        n = 30
        r = range(4)
        curr = [np.zeros(n) for _ in r]
        printed_stall = [True for _ in r]
        printed_not_stall = [True for _ in r]
        
        mean_stalledness = [[] for _ in r]
        stalled = [0 for _ in r]
        
        t = []
        y_lists = [[] for _ in r]
        
        s = 0.5
        motors[0].set_speed(s)
        motors[1].set_speed(s)
        motors[2].set_speed(-s)
        motors[3].set_speed(-s)
        i = 0
        while True:
            t.append(i)
            for j in r:
                data = motors[j].read()
                c = abs(data[1])
                curr[j][i % n] = c
                deltaCurr = np.median(curr[j]) - c
                if i < n: continue
                mean_stalledness[j].append(np.mean(curr[j]))
                stalled[j] = 0
                if len(mean_stalledness[j]) > n:
                    if mean_stalledness[j][-1] <= np.median(mean_stalledness[j]) - 0.02 * np.max(mean_stalledness[j]):
                        stalled[j] = 1
                y_lists[j].append(deltaCurr)
                # ~ if deltaCurr > 0.85 * pow(10, 7):
                    # ~ printed_not_stall[j] = False
                    # ~ if not printed_stall[j]:
                        # ~ print(j, "is stalled.")
                        # ~ printed_stall[j] = True
                # ~ else:
                    # ~ printed_stall[j] = False
                    # ~ if not printed_not_stall[j]:
                        # ~ print(j, "NOT stalled.")
                        # ~ printed_not_stall[j] = True
            if sum(stalled) >= 3:
                print("STALLED", stalled)
            i += 1
            sleep(0.001)
            
    except KeyboardInterrupt:
        [motors[j].set_speed(0) for j in range(4)]
        
        for j in r:
            y = y_lists[j] / np.max(y_lists[j])
            y = np.power(y, 3)
            plt.plot(t[:len(y_lists[j])], y)
            plt.plot(t[:len(y_lists[j])], mean_stalledness[j] / np.max(mean_stalledness[j]))
            
        plt.xlabel("time")
        plt.ylabel("current")
        plt.title("bingus")
    
        plt.show()
    

s = 1.0
motors[0].set_speed(s)
motors[1].set_speed(s)
motors[2].set_speed(-s)
motors[3].set_speed(-s)

sleep(1)
[motors[j].set_speed(0) for j in range(4)]
