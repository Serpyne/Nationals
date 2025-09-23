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

    N = 300

    t = np.zeros(N)
    y_lists = [np.zeros(N, dtype=np.float64) for _ in range(4)]
    medians = [np.zeros(N, dtype=np.float64) for _ in range(4)]
    
    sample_n = 25
    target_speed = 1.0

    initial_t = now()
    for i in range(N):
        min_median = max([medians[j][i - 1] for j in range(4)])
        set_value = target_speed if i <= sample_n + 1 else target_speed * medians[j][i - 1] / min_median
        s = np.array([1, -1, -1, 1]) * set_value
        [motors[j].set_speed(s[j]) for j in range(4)]
        
        t[i] = now() - initial_t
        for j in range(4):
            data = motors[j].read()
            y_lists[j][i] = abs(data[1])
            if i >= sample_n:
                medians[j][i] = np.median(y_lists[j][i-sample_n : i])
        sleep(0.0001)

    [motors[j].set_speed(0) for j in range(4)]

    colours = ["red", "yellow", "green", "blue"]
    for j in range(4):
        c = colours[j]
        plt.plot(t, y_lists[j], color=c, alpha=0.1)
        plt.plot(t, medians[j], color=c)
        
        # ~ median_value = np.median(y_lists[j])
        # ~ plt.axhline(median_value, color=c, linestyle='--', label=f'Median: {median_value:.2f}')

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

def plot_speed():

    N = 300

    t = np.zeros(N)
    speeds = [np.zeros(N, dtype=np.float64) for _ in range(4)]
    means = [np.zeros(N, dtype=np.float64) for _ in range(4)]
    
    target_speed = 0.5
    s = np.array([1, 1, -1, -1]) * target_speed
    [motors[j].set_speed(s[j]) for j in range(4)]
    
    num_samples = 15

    initial_t = now()
    for i in range(N):
        
        t[i] = now() - initial_t
        for j in range(4):
            data = motors[j].read()
            
            desired = target_speed
            error = desired - abs(data[1]) / motors[j].max_speed
            speeds[j][i] = error
            
            mean_speed = np.mean(speeds[j][i - (num_samples + 1) : i - 1]) if i > num_samples + 2 else 0
            means[j][i] = mean_speed
            
        sleep(0.0001)

    [motors[j].set_speed(0) for j in range(4)]
    
    first_nonzero = 0
    # ~ for i in range(N):
        # ~ if means[0][i] > 4.5 * pow(10, 7):
            # ~ first_nonzero = i
            # ~ break

    colours = ["red", "yellow", "green", "blue"]
    for j in range(4):
        c = colours[j]
        plt.plot(t[first_nonzero:], speeds[j][first_nonzero:], color=c, alpha=0.1)
        plt.plot(t[first_nonzero:], means[j][first_nonzero:], color=c)
        

    plt.xlabel("time")
    plt.ylabel("current")
    plt.title("bingus")

    plt.show()

# ~ plot_speed()

def plot_position():

    N = 300

    t = np.zeros(N)
    positions = [np.zeros(N, dtype=np.float64) for _ in range(4)]
    means = [np.zeros(N, dtype=np.float64) for _ in range(4)]
    
    target_speed = 0.9
    s = np.array([1, 1, -1, -1]) * target_speed
    [motors[j].set_speed(s[j]) for j in range(4)]
    
    num_samples = 15

    initial_t = now()
    for i in range(N):
        
        t[i] = now() - initial_t
        for j in range(4):
            data = motors[j].read()
            
            if i == 0: continue
            positions[j][i] = data[1] - positions[j][i-1]
            
            mean_speed = np.mean(positions[j][i - (num_samples + 1) : i - 1]) if i > num_samples + 2 else 0
            means[j][i] = mean_speed
            
        sleep(0.0001)

    [motors[j].set_speed(0) for j in range(4)]
    
    colours = ["red", "yellow", "green", "blue"]
    for j in range(4):
        c = colours[j]
        plt.plot(t, positions[j], color=c, alpha=0.1)
        plt.plot(t, means[j], color=c)
        

    plt.xlabel("time")
    plt.ylabel("current")
    plt.title("bingus")

    plt.show()

plot_position()

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.previous_error = 0

    def update(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output
        
        
def pid_test():
    N = 1200

    t = np.zeros(N)
    speed = np.zeros(N, dtype=np.float64)
    control = np.zeros(N, dtype=np.float64)
    
    pid = PIDController(Kp=0.001, Ki=0.00, Kd=0.0001)
    
    target_speed = 0.8
    current_speed = target_speed
    
    initial_t = now()
    prev = initial_t
    for i in range(N):
        
        motors[0].set_speed(current_speed)
        
        curr = now()
        t[i] = curr - initial_t
        dt = curr - prev
        prev = curr
        
        data = motors[0].read()
        measured_speed = abs(data[1]) / motors[0].max_speed
        speed[i] = measured_speed
        
        control_signal = pid.update(target_speed, measured_speed, dt)
        
        current_speed = max(-1, min(1, current_speed + control_signal))
        control[i] = control_signal
            
        sleep(0.001)

    motors[0].set_speed(0)
    
    for j in range(4):
        plt.plot(t, speed, color="red", alpha=0.2)
        plt.plot(t, control, color="green")
        
    plt.xlim(0, t[-1])
    plt.ylim(0, 1)
    plt.xlabel("time")
    plt.ylabel("current")
    plt.title("bingus")

    plt.show()

# ~ pid_test()
