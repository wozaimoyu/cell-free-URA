from Network import Network
import time
import numpy as np
from utilities import computeNoisePower, QPSK,scatterPlot
from UserEquipment import UEs
from AccesPoints import APs
import matplotlib.pyplot as plt
np.random.seed(0)

# ====================================== Initialization ====================================== #
# --- Simulation
nIter = 100

# --- Wireless Network
D = 1000
nAPs = 16
nAnts = 1
nUEs = 3
fc = 1.9e9
BW = 20e6

# --- Noise
noiseFiguredB = 7
sigma2 = computeNoisePower(BW, noiseFiguredB)

# --- Create a network
net = Network(D, nAPs, nAnts, nUEs, fc)

# --- Coding Scheme
# number of pilots
nPilots = 1024# 640, 896
# Spreading sequence length
L = 9
# Length of the code
nc = 512
# Length of the message, Length of the First and Second part of the message
B, Bf = 100, 10
Bs = B - Bf
# Number of spreading sequence/ pilot sequence
J = 2 ** Bf
# Length of List (Decoder)
nL = 64
# Number of channel uses
nChanlUses = int((nc / np.log2(4))*L + nPilots)
# User transmit power
Pt = 100e-3# 100mW

probDE = np.zeros(nIter)
probFA = np.zeros(nIter)
evolution = np.zeros(nIter)
# ====================================== Simulation ====================================== #
start = time.time()
print("# ============= Simulation ============= #")
print('Number of users: ' + str(nUEs))
print('Number of Access Points: ' + str(nAPs))
print("Number of channel uses: " +str(nChanlUses))
print("Number of iterations: "+ str(nIter))
print("Length of spreading sequence " + str(L))
print("Length of the code " +str(nc))
print("Length of message bits = " +str(B))
print("Bf = " + str(Bf))
print('Transmit Power (Pt): ' + str(Pt*1000) + 'mW')


error = 0
for Iter in range(nIter):

    # --- Generate Users at Random
    net.generateUsers()
    if 0:
        net.plotCell()

    # --- Generate nUEs msgs
    msgs = np.random.randint(0, 2, (nUEs, B))

    # --- Generate the channel coefficients
    H = net.generateChannel()

    # --- Generate the noise
    N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2), (nChanlUses, nAPs * nAnts)) + 1j * np.random.normal(0, np.sqrt(sigma2), (nChanlUses, nAPs * nAnts)))

    # --- Create an User Equipment object
    users = UEs(nUEs, nAPs, nAnts, B, Bf, L, nc, nL, nPilots, sigma2, Pt)

    # H = np.ones((nUEs, nAPs))
    # --- Transmit data
    HX = users.transmit(msgs, H)

    Y = HX + N

    # --- Create an Access Points object
    accessPoints = APs(nUEs, nAPs, nAnts, B, Bf, L, nc, nL, nPilots, sigma2, users)

    # --- APs receiver
    nDE, nFA = accessPoints.receiver(Y)

    probFA[Iter] = nFA
    probDE[Iter] = nDE

    if Iter == 0:
        evolution[Iter] = probFA[Iter] + 1 - probDE[Iter]
    else:
        evolution[Iter] = evolution[Iter - 1] + probFA[Iter] + 1 - probDE[Iter]


iterRange = np.arange(nIter) + 1
evolution = evolution / iterRange

print()


print()