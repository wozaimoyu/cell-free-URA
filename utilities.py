import numpy as np
from scipy.constants import Boltzmann
import matplotlib.pyplot as plt

def computeNoisePower(BW, noiseFiguredB):
    T0 = 290
    noiseFigure = 10 ** (noiseFiguredB / 10)
    return BW * Boltzmann * T0 * noiseFigure

def scatterPlot(signal,xLimits=None,yLimits=None,title=None):
    real = np.real(signal)
    imag = np.imag(signal)

    plt.scatter(real, imag)
    plt.grid()
    if xLimits!=None:
        plt.xlim([xLimits[0], xLimits[1]])
    if yLimits!=None:
        plt.xlim([yLimits[0], yLimits[1]])

    plt.xlabel('Q')
    plt.ylabel('I')
    plt.title(title)
    plt.show()

# === Convert an array of integers to a 2D array of binary Strings === #
def dec2bin(listDec, nBits):
    # From decimal to string(binary) array
    binary_repr_v = np.vectorize(np.binary_repr)
    binaryString = binary_repr_v(listDec, width=nBits)

    # Number of integers
    n = listDec.shape
    if (len(n) != 0):
        n = len(listDec)
    else:
        n = 1
        # Split the string
        binStr = list(map(int, binaryString))

        # Save the binary stream
        return np.real(binStr)

    # Define a 2D array to save the binary streams
    mtxBinary = np.zeros((n, nBits))

    # For each string, break the string and make it array int
    for i in range(n):
        # Take the next string
        binStr = binaryString[i]

        # Split the string
        binStr = list(map(int, binStr))

        # Save the binary stream
        mtxBinary[i] = np.real(binStr)

    return mtxBinary


# === Convert  a 2D array of binary Strings to an array of integers === #
def bin2dec(listBin):
    # Number of bits
    bits = listBin.shape[0]

    # Define a 2D array to save the binary streams
    temp = np.flip(2 ** np.arange(bits))
    intNum = np.dot(listBin, temp)

    return int(intNum)


# ========================= CRC Encoder/Decoder ======================= #

# ---- Encoder
def crcEncoder(dividend, divisor):
    # ======
    n1, n2 = len(dividend), len(divisor)

    # --- Append n2 zeros to dividend
    newDividend = np.append(dividend, np.zeros(n2, dtype=int)).astype(int)

    # --- Compute the reminder
    i = 0
    while (1):
        # XOR the first n1 bits

        newDividend[i:i + n2] = newDividend[i:i + n2] ^ divisor

        while (newDividend[i] == 0 and i <= n1):
            i += 1
        if (i > n1):
            break

    # --- Append the remider to the dividend
    crcCodeword = np.append(dividend, newDividend[n1:n1 + n2])
    return crcCodeword


def crcDecoder(dividend, divisor):
    # ======
    n1, n2 = len(dividend), len(divisor)
    dividend = dividend.astype(int)

    # --- If the estimate msg is the all zero msg
    if (sum(dividend) == 0):
        return 1
    else:
        ones = np.nonzero(dividend)
        # --- If the degree of the message is less than the degree of the divisor
        # The msg is wrong
        if (ones[0][0] > n1 - n2):
            return 0

    # --- If the msg doesn't belong to one of these special cases, decode
    if (ones[0][0] > 0):
        dividend = dividend[ones[0][0]::]
    # --- Compute the reminder
    i = 0
    while (1):
        # XOR the first n1 bits
        if (len(dividend[i:i + n2]) != len(divisor)):
            return 0
        dividend[i:i + n2] = dividend[i:i + n2] ^ divisor

        if (sum(dividend) == 0):
            return 1

        if (dividend[i] == 1):
            # --- Don't Shift
            continue
        else:
            # --- Find the next 1
            ones = np.nonzero(dividend)
            # --- Shift
            i = ones[0][0]

            if (len(dividend[i::]) < n2):
                # Failure
                return 0


def modQPSK(data):
    # Reshape data
    data = np.reshape(data, (2, -1))
    symbols = 1.0 - 2.0 * data
    return np.sqrt(0.5) * (symbols[0, :] + 1j * symbols[1, :])

def demQPSK(symbols):
    # Reshape data
    data = np.zeros(len(symbols) * 2)
    realPart = (np.real(symbols) < 0 ) * 1
    imagPart = (np.imag(symbols) < 0) * 1
    data[0:len(symbols)] = realPart
    data[len(symbols)::] = imagPart
    return data

def LMMSE(y, A, Qx, Qn):
    '''
        Function that implements the LMMSE Traditional and Compact Form estimator
        for the system y = Ax + n

        Input: y: received vector, A: measurement matrix, Qx: covariance of x, Qn: covariance of the noise
        Output: xHat: estimate of x, MSE: covariance matrix of the error
    '''

    r, c = A.shape

    # ========================= Traditional LMMSE ========================= #
    # https://en.wikipedia.org/wiki/Minimum_mean_square_error
    if r <= c:

        # Covariance Cov(X,Y)
        Qxy = np.dot(Qx, A.conj().T)

        # Covariance of Y
        Qy = np.dot(np.dot(A, Qx), A.conj().T) + Qn

        # Inverse of Covariance of Y
        QyInv = np.linalg.inv(Qy)

        # --- Filter
        F = np.dot(Qxy, QyInv)


    # ========================= Compact LMMSE ========================= #
    else:
        # --- Inverse of matrices
        QxInv = np.linalg.inv(Qx)

        QnInv = np.linalg.inv(Qn)

        # --- Second Term
        W2 = np.dot(A.conj().T, QnInv)

        # --- First term
        W1 = QxInv + np.dot(W2, A)
        W1Inv = np.linalg.inv(W1)

        # --- Filter
        F = np.dot(W1Inv, W2)

    # --- Estimates
    xHat = np.dot(F, y)

    return xHat

