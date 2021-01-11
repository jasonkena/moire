import torch
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def series_cn(y, n, time, period):
    # [n, time]
    time = time.unsqueeze(0)
    y = y.unsqueeze(0)
    n = n.unsqueeze(1)

    c = y * torch.exp(-1j * 2 * n * np.pi * time / period)
    c = torch.sum(c, dim=1) / c.size(1)

    return c


def series_forward(c, n, time, period):
    # [n, time]
    time = time.unsqueeze(0)
    c = c.unsqueeze(1)
    n = n.unsqueeze(1)

    y = c * np.exp(1j * 2 * n * np.pi * time / period)
    y = torch.sum(y, dim=0)
    return y


def square_wave():
    t = torch.linspace(0, 2*np.pi-1e-5, 500)
    y=torch.from_numpy(signal.square(t))

    return y,t

def load_data(file_name):
    data = pd.read_csv(os.path.join("data", file_name), delim_whitespace=True).to_numpy()
    data = torch.from_numpy(data)

    t = data[:,0]
    y = data[:,1]
    interval = t[1] - t[0]

    return y, t, interval

def select_period(y,t,period):
    is_valid = t<period
    y = y[is_valid]
    t = t[is_valid]

    return y,t

period = np.deg2rad(120)
y, t, interval = load_data("no_interlayer.raw")
y,t = select_period(y,t,period)

# period = 2*np.pi
# y,t = square_wave()
# Nyquist
n = (period/interval)//2
n = torch.arange(-n,n+1)
# # n = torch.arange(0,n+1)

c = series_cn(y,n,t,period)

# # t_test = t
t_test = torch.linspace(-1,np.pi, int(1e3))
y_reconstruct = series_forward(c,n,t_test,period)

# plt.plot(t_test,y_reconstruct)
# plt.plot(t,y)
plt.plot(n,c.abs())
plt.show()
