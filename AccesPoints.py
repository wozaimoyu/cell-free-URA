import numpy as np
from PolarCode import PolarCode
from utilities import bin2dec, dec2bin, crcEncoder, crcDecoder, QPSK, LMMSE
from estiFuncs import symbolsEst, channelEst, channelEstWithErrors
import matplotlib.pyplot as plt

class APs:
    def __init__(self, nUEs, nAPs, nAnts, B, Bf, L, nc, nL, nPilots, sigma2, UEs):
        ''' Parameters '''
        self.nUEs = nUEs  # Number of Users
        self.nAPs = nAPs  # Number of APs
        self.nAnts = nAnts # Number of antennas per AP
        self.totalAnt = self.nAnts * self.nAPs
        self.Bf = Bf  # number of bits of the first part of the message
        self.Bs = B - Bf  # number of bits of the second part of the message
        self.L = L  # Length of spreading sequence
        self.J = 2 ** Bf  # Number of spreading sequence
        self.nc = nc  # length of code
        self.nL = nL  # List size
        self.nQPSKSymbols = int(nc / 2)  # Number of QPSK symbols
        self.nDataSymbols = int(L * self.nQPSKSymbols)
        self.nPilots = nPilots  # number of pilot symbols
        self.nChanlUses = self.nPilots + self.nDataSymbols
        self.sigma2 = sigma2
        self.UEs = UEs
        self.N = int(nUEs * 1)
        self.save = 0

        ''' For polar code '''
        self.divisor = self.UEs.divisor


        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.msgLen = self.Bs + self.lCRC  # Length of the input to the encoder
        self.frozenValues = self.UEs.frozenValues

        # Create a polar Code object
        self.polar = PolarCode(self.nc, self.msgLen, self.nUEs)

        ''' Generate matrices '''
        # Pilots
        self.P = self.UEs.P

        # Spreading sequence master set
        self.A = self.UEs.A


        # Interleaver
        self.interleaver = self.UEs.interleaver

        ''' To store information '''
        self.msgs = np.zeros((nUEs, Bf + self.Bs), dtype=int)  # Store the active messages
        self.msgsHat = np.zeros((nUEs, Bf + self.Bs), dtype=int)  # Store the recovered messages

        self.idxSSHat = np.zeros((self.nAPs, self.N), dtype=int)

        self.count = 0  # Count the number of recovered msgs in this round
        self.Y = np.zeros((self.nChanlUses, nAPs))
        self.idxSSDec = np.array([], dtype=int)
        # self.idxSSHat = np.array([], dtype=int)  # To store the new recovered sequences
        self.symbolsHat = np.zeros((self.nUEs, self.nQPSKSymbols), dtype=complex)


    def receiver(self, Y):
        # === Step 1: Energy Detector
        self.idxSSHat = energyDetector(self, Y, self.N)
        # Check performance
        nDE, nFA, nTotalDE = checkEnergyDetector(self)

        # return sum(nDE)/self.nAPs, sum(nFA) / self.nAPs
        print('Detection prob per AP = ' + str(sum(nDE)/self.nAPs))
        print('False Alarm prob per AP = ' + str(sum(nFA) / self.nAPs))
        print('Probability of Detection = ' + str(nTotalDE))
        print()

# ============================================ Functions ============================================ #
# === Energy Detector
def energyDetector(self, y, nUEs):
    # --- Energy Per Antenna
    energy = abs(np.dot(self.P.conj().T, y[0:self.nPilots, :])) ** 2

    pivot = self.nPilots
    for t in range(self.nQPSKSymbols):
        energy += abs(np.dot(self.A[t * self.L: (t + 1) * self.L, :].conj().T, y[pivot + t * self.L: pivot + (t + 1) * self.L, :])) ** 2

    # Combine only the energy from the same APs
    energy = np.sum(np.reshape(energy, (energy.shape[0], self.nAPs, self.nAnts)), axis=2)
    # print(np.sort(self.UEs.idxSS))
    # print()

    idxSSHat = np.zeros((self.nAPs, self.N), dtype=int)
    for i in range(self.nAPs):
        # print(i)
        # print(np.sort((-energy[:, i]).argsort()[:nUEs]))
        idxSSHat[i, :] = (-energy[:, i]).argsort()[:nUEs]
    return idxSSHat


def checkEnergyDetector(self):
    nDE = np.zeros(self.nAPs)
    nFA = np.zeros(self.nAPs)
    nTotalDE = 0
    correctIdx = []


    for i in range(self.nAPs):
        for j in range(self.N):
            if self.idxSSHat[i, j] in self.UEs.idxSS:
                nDE[i] += 1
                if self.idxSSHat[i, j] not in correctIdx:
                    correctIdx.append(self.idxSSHat[i, j])
                    nTotalDE += 1
                # else:

            else:
                nFA[i] += 1
    nDE /= self.N
    nFA /= self.N
    nTotalDE /= self.nUEs


    return nDE, nFA, nTotalDE