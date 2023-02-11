import numpy as np
from PolarCode import PolarCode
from utilities import demQPSK, dec2bin, crcDecoder, scatterPlot
import matplotlib.pyplot as plt


class CPU:
    def __init__(self, nUEs, nAPs, B, Bf, nc, nL, sigma2, debug):
        ''' Parameters '''
        self.nUEs = nUEs  # Number of Users
        self.nAPs = nAPs  # Number of APs
        self.B = B
        self.Bf = Bf  # number of bits of the first part of the message
        self.Bs = B - Bf  # number of bits of the second part of the message
        self.nc = nc  # length of code
        self.nL = nL  # List size
        self.nQPSKSymbols = int(nc / 2)  # Number of QPSK symbols
        self.sigma2 = sigma2

        ''' For polar code '''
        # Polynomial for CRC coding
        if nUEs < 10:
            self.divisor = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)
        else:
            self.divisor = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1], dtype=int)

        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.msgLen = self.Bs + self.lCRC  # Length of the input to the encoder

        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.msgLen = self.Bs + self.lCRC  # Length of the input to the encoder
        self.debug = debug
        self.msgsHat = np.zeros((nUEs, B))

    def combine(self, symbolsHat, idxSSHat, interleaver, frozenvalues, messages, idx2UE, codewordInter):

        self.count = 0
        uniqIdx = np.unique(np.ndarray.flatten(idxSSHat))
        self.msgsHat = np.zeros((len(uniqIdx), self.B))
        # Create a polar code object
        polar = PolarCode(self.nc, self.msgLen, 1)

        # --- Use the index of the pilot to combine the signals
        for ue, actIdx in enumerate(uniqIdx):
            # Find which of the APs have symbols for this user (Assuming no collisions)
            apIdx, ueIdx = np.where(actIdx == idxSSHat)

            # Combine the symbols
            currSymb = sum(symbolsHat[ueIdx, apIdx, :])

            if self.debug:
                UsersTemp= idx2UE[actIdx]
                print('BER (' + str(UsersTemp) + ') = ' + str(np.sum(
                    np.mod(codewordInter[UsersTemp, :] + demQPSK(currSymb), 2)) / self.nc))

            # Call Polar Decoder
            msgHat, isDecoded = channDecoder(currSymb, polar, interleaver[:, actIdx], frozenvalues[:, actIdx], self.nL, self.divisor)
            if isDecoded:

                firsPart = dec2bin(np.array([actIdx]), self.Bf)[0]
                self.msgsHat[self.count, :] = np.concatenate((firsPart, msgHat[0:self.Bs]), 0)
                self.count += 1
        nDE, nFA = checkPerformance(self, messages)


        return nDE, nFA, self.count

def channDecoder(symbolsHat, polar, interleaver, frozenValues, nL, divisor):

        # Demodulate the QPSK symbols
        cwordHatSoft = np.concatenate((np.real(symbolsHat), np.imag(symbolsHat)), 0)
        # Interleaver
        cwordHatSoftInt = np.zeros(polar.n)
        cwordHatSoftInt[interleaver] = cwordHatSoft

        # ============ Polar decoder ============ #
        msgCRCHat, PML = polar.listDecoder(np.sqrt(2) * cwordHatSoftInt, frozenValues, nL)

        # ============ Check CRC ============ #
        # --- Initialization
        thres, flag = np.Inf, -1

        # --- Check the CRC constraint for all message in the list
        for l in range(nL):
            check = crcDecoder(msgCRCHat[l, :], divisor)
            if check:
                # --- Check if its PML is larger than the current PML
                if PML[l] < thres:
                    flag = l
                    thres = PML[l]

        # --- Encode the estimated message
        if thres != np.Inf:
            msg2Hat = msgCRCHat[flag, :]
            isDecoded = 1
        else:
            msg2Hat = 0
            isDecoded = 0


        return msg2Hat, isDecoded


def checkPerformance(self, msgs):
    nDE, nFA = 0, 0
    for i in range(self.count):
        flag = 0
        for k in range(self.nUEs):
            binSum = sum((msgs[k, :] + self.msgsHat[i, :]) % 2)

            if binSum == 0:
                flag = 1
                break
        if flag == 1:
            nDE += 1
        else:
            nFA += 1

    return nDE, nFA