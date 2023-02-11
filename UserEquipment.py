# -*- coding: utf-8 -*-
import numpy as np
from PolarCode import PolarCode
from utilities import bin2dec, dec2bin, crcEncoder, crcDecoder, modQPSK, LMMSE, demQPSK
from estiFuncs import symbolsEst, channelEst, channelEstWithErrors
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

class UEs:
    def __init__(self, nUEs, nAPs, nAnts, B, Bf, L, nc, nL, nPilots, sigma2, Pt, timeVarying):
        ''' Parameters '''
        self.nUEs = nUEs  # Number of Users
        self.nAPs = nAPs  # Number of APs
        self.nAnts = nAnts # Number of antennas per APs
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
        self.Pt = Pt
        self.save = 0
        self.timeVarying = timeVarying

        ''' For polar code '''
        # Polynomial for CRC coding
        if nUEs < 10:
            self.divisor = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1], dtype=int)
        else:
            self.divisor = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1], dtype=int)

        self.lCRC = len(self.divisor)  # Number of CRC bits
        self.msgLen = self.Bs + self.lCRC  # Length of the input to the encoder
        self.frozenValues = np.round(np.random.randint(low=0, high=2, size=(self.nc - self.msgLen, self.J)))

        # Create a polar Code object
        self.polar = PolarCode(self.nc, self.msgLen, self.nUEs)

        ''' Generate matrices '''
        # Pilots
        self.P = np.sqrt(self.Pt/2) * ((1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.nPilots, self.J)))) + 1j * (
                1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.nPilots, self.J)))))

        # self.P = (hadamard(self.J) + 1j*hadamard(self.J)) / np.sqrt(2.0)
        # Spreading sequence master set
        if self.timeVarying == 1:
            self.A = (np.random.normal(loc=0, scale=1, size=(self.nQPSKSymbols * self.L, self.J)) + 1j * np.random.normal(
                loc=0, scale=1, size=(self.nQPSKSymbols * self.L, self.J)))

            for j in range(self.nQPSKSymbols):
                temp = np.linalg.norm(self.A[j * self.L:(j + 1) * self.L, :], axis=0)
                self.A[j * self.L:(j + 1) * self.L, :] = np.divide(self.A[j * self.L:(j + 1) * self.L, :], temp)

            self.A = (np.sqrt(self.L) * self.A) * np.sqrt(self.Pt)
        else:
            self.A = (np.random.normal(loc=0, scale=1, size=(self.L, self.J)) + 1j * np.random.normal(loc=0, scale=1, size=(self.L, self.J)))

            temp = np.linalg.norm(self.A, axis=0)
            self.A = np.divide(self.A, temp)

            self.A = (np.sqrt(self.L) * self.A) * np.sqrt(self.Pt)
            # self.A = np.sqrt(self.Pt / 2) * ((1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.L, self.J)))) + 1j * (
            #             1 - 2 * np.round(np.random.randint(low=0, high=2, size=(self.L, self.J)))))

        # Interleaver
        self.interleaver = np.zeros((self.nc, self.J), dtype=int)
        for j in range(self.J):
            self.interleaver[:, j] = np.random.choice(self.nc, self.nc, replace=False)

        ''' To store information '''
        self.msgs = np.zeros((nUEs, Bf + self.Bs), dtype=int)  # Store the active messages
        self.msgsHat = np.zeros((nUEs, Bf + self.Bs), dtype=int)  # Store the recovered messages

        self.idxSS = np.zeros(self.nUEs, dtype=int)

        self.count = 0  # Count the number of recovered msgs in this round
        self.Y = np.zeros((self.nChanlUses, nAPs))
        self.idxSSDec = np.array([], dtype=int)
        self.idxSSHat = np.array([], dtype=int)  # To store the new recovered sequences
        self.symbolsHat = np.zeros((self.nUEs, self.nQPSKSymbols), dtype=complex)
        self.codewordInter = np.zeros((self.nUEs, self.nc))

        self.idx2UE = np.zeros(self.J, dtype=int) - 1
        self.UE2idx = np.zeros(self.nUEs, dtype=int) - 1


    def transmit(self, msgs, H):

        '''
        Function to encode the messages of the users
        Inputs: 1. the message of the users in the binary form, dimensions of msgBin, K x B
                2. Channel
        Output: The sum of the channel output before noise, dimensions of Y, n x nAPs
        '''

        # ===================== Initialization ===================== #
        HX = np.zeros((self.nChanlUses, self.totalAnt), dtype=complex)



        firstPart = np.random.choice(np.arange(self.J), self.nUEs, replace=False)
        msgs[:,0:self.Bf] = dec2bin(firstPart, self.Bf)




        # --- Step 0: Save the messages
        self.msgs = msgs.copy()
        # --- For all active users
        for k in range(self.nUEs):

            # --- Step 1: Break the message into two parts
            # First part, Second part
            mf = self.msgs[k, 0:self.Bf]
            ms = self.msgs[k, self.Bf::]

            # --- Step 2: Find the decimal representation of mf
            self.idxSS[k] = bin2dec(mf)
            self.idx2UE[self.idxSS[k]] = int(k)
            self.UE2idx[k] = int(self.idxSS[k])

            # --- Step 3: Append CRC bits to ms
            msgCRC = crcEncoder(ms, self.divisor)

            # --- Step 4: polar encode
            codeword, _ = self.polar.encoder(msgCRC, self.frozenValues[:, self.idxSS[k]], k)

            # --- Step 5: Interleaver
            codeword = codeword[self.interleaver[:, self.idxSS[k]]]

            self.codewordInter[k,:] = codeword
            # --- Step 6: QPSK modulation
            symbols = modQPSK(codeword)


            # --- For Pilots (PH)
            PH = np.kron(self.P[:, self.idxSS[k]], H[k, :]).reshape(self.nPilots, self.totalAnt)

            # --- For Symbols (QH)
            if self.timeVarying == 1:
                A = np.zeros((self.nDataSymbols), dtype=complex)
                for t in range(self.nQPSKSymbols):
                    A[t * self.L: (t + 1) * self.L] = self.A[t * self.L: (t + 1) * self.L, self.idxSS[k]] * symbols[t]
            else:
                A = np.kron(symbols, self.A[:, self.idxSS[k]])

            QH = np.kron(A, H[k, :]).reshape(self.nDataSymbols, self.totalAnt)

            # --- Add the new matrix to the output signal
            HX += np.vstack((PH, QH))

        # HX3 = np.reshape(HX2, (self.nChanlUses, self.nAPs, self.nAnts))

        return HX







