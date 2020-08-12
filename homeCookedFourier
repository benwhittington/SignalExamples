import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

def fourier(f_x):
  N = f_x.shape[0]
  out = np.zeros(N)

  for k in range(N):
    for n in range(N):
      out[k] += f_x[n] * np.exp((-1j * 2 * np.pi * k * n) / N)

  return out

def ft_demirror(f_x):
  N = f_x.shape[0]
  N_out = int(N // 2 + (N / 2 - N // 2) * 2)  # add one more if odd
  out = np.zeros(N_out)

  for i in range(N_out):
    out[i] = f_x[i] + f_x[-i]

  return out

def f1(xt, *args):
  return np.array([
    0 if x < -0.5 or x > 0.5
    else 0.5 if x == -0.5 or x == 0.5
    else 1 for x in xt
  ])

def f0(x, *args):
  return np.sin(x)

def f_arb(x, freqs):
  out = np.zeros(x.shape[0])

  for f in freqs:
    out += np.sin(2 * np.pi * f * x)

  return out

def main():
  N = 1000
  x = np.linspace(0, 1 * np.pi, N)

  f = f1

  args = [1, 2, 10]

  f_x = f(x, args)

  _, ax = plt.subplots(3)

  ax[0].plot(x, f_x)
  ax[0].set_title("f_x")

  ax[1].plot(ft_demirror(np.abs(fourier(f_x))))
  ax[1].set_title("Cust Fourier")

  ax[2].plot(ft_demirror(np.abs(fft(f_x))))
  ax[2].set_title("np.fft")

  plt.show()

if __name__ == "__main__":
  main()