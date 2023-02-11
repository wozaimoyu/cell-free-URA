from Network import Network
import time
import numpy as np
from utilities import computeNoisePower
from CPU import CPU
from UserEquipment import UEs
from AccesPoints import APs
import matplotlib.pyplot as plt
np.random.seed(0)

# ====================================== Initialization ====================================== #
# --- Simulation
nIter = 100
debug = 0
# --- Wireless Network
D = 1000
nAPs = 64 #(nAPs/nUEs = 10)
nAnts = 1
nUEs = 50
BW = 125e3
nRec = 4


# --- Noise
noiseFiguredB = 9
sigma2 = computeNoisePower(BW, noiseFiguredB)

# --- Create a network
net = Network(D, nAPs, nAnts, nUEs)

# --- Coding Scheme
# number of pilots
nPilots = 896# 640, 896
# Spreading sequence length
L = 9
timeVarying = 0
# Length of the code
nc = 512
# Length of the message, Length of the First and Second part of the message
B, Bf = 100, 12
Bs = B - Bf
# Number of spreading sequence/ pilot sequence
J = 2 ** Bf
# Length of List (Decoder)
nL = 8
# Number of channel uses
nChanlUses = int((nc / np.log2(4))*L + nPilots)
# User transmit power
Pt = 2e-3# 100mW

cpu = CPU(nUEs, nAPs, B, Bf, nc, nL, sigma2, debug)

probDE = np.zeros(nIter)
probFA = np.zeros(nIter)
probDEperAP = np.zeros((nAPs, nIter))
probFAperAP = np.zeros((nAPs, nIter))
totalDet = np.zeros(nIter)
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
print('Tx SNR (int): ' + str(int(10*np.log10(Pt/sigma2))) + 'dB')
Llinear = np.sqrt(10 ** (net.L/10))
print('Max Rx SNR (int): ' + str(int((10*np.log10(Pt*Llinear/sigma2)))) + 'dB')
print('Tx power : ' + str(Pt) + 'W')

error = 0
for Iter in range(nIter):

    # --- Generate Users at Random
    net.generateUsers()
    if 1 and Iter == 50:
        net.plotCell()

    # --- Generate nUEs msgs
    msgs = np.random.randint(0, 2, (nUEs, B))

    # --- Generate the channel coefficients
    H = net.generateChannel()

    # --- Generate the noise
    N = (1 / np.sqrt(2)) * (np.random.normal(0, np.sqrt(sigma2), (nChanlUses, nAPs * nAnts)) + 1j * np.random.normal(0, np.sqrt(sigma2), (nChanlUses, nAPs * nAnts)))

    # --- Create an User Equipment object
    users = UEs(nUEs, nAPs, nAnts, B, Bf, L, nc, nL, nPilots, sigma2, Pt, timeVarying)

    # H = np.ones((nUEs, nAPs))
    # --- Transmit data
    HX = users.transmit(msgs, H)

    Y = HX + N

    # --- Create an Access Points object
    accessPoints = APs(nUEs, nAPs, nAnts, B, Bf, L, nc, nL, nPilots, sigma2, users, nRec, net, debug)

    # --- APs receiver
    symbolsHat, idxSSHat = accessPoints.receiver(Y,H)

    # --- CPU
    nDE, nFA, recnUEs = cpu.combine(symbolsHat, idxSSHat, users.interleaver, users.frozenValues,  users.msgs, users.idx2UE, users.codewordInter)
    probFA[Iter] = nFA/recnUEs
    probDE[Iter] = nDE/nUEs

    if Iter == 0:
        evolution[Iter] = probDE[Iter]
    else:
        evolution[Iter] = evolution[Iter - 1] + probDE[Iter]


iterRange = np.arange(nIter) + 1
evolution = evolution / iterRange

plt.plot(iterRange,evolution)
plt.title('Evolution of the Detection probability')
plt.xlabel('Realization #')
plt.ylabel('Probability of Detection')
plt.show()
print(sum(probFA)/nIter)
print(sum(probDE)/nIter)
print()


print()