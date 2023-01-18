import numpy as np
import matplotlib.pyplot as plt




class Network:
    def __init__(self, D, nAPs, nAnts, nUEs, fc):
        self.D = D
        self.nAPs = nAPs
        self.nAnts = nAnts # Number of antennas per AP
        self.nUEs = nUEs
        self.fc = fc
        self.UEsCoordinates = np.zeros((2, self.nUEs))
        self.ha = 15
        self.hu = 1.65
        self.d0 = 10
        self.d1 = 50
        self.dDecorr = 0.1e3
        self.delta = 0.8
        self.L = computeL(self.fc, self.ha, self.hu)
        self.computeCentroids()
        self.distAP2UE = np.zeros((self.nUEs, self.nAPs))
        self.distUE2UE = np.zeros((self.nUEs, self.nUEs))
        self.distAP2AP = np.zeros((self.nAPs, self.nAPs))
        self.pathLoss = np.zeros((self.nUEs, self.nAPs))
        self.shadowingCovUEs = np.zeros((self.nUEs, self.nUEs))
        self.shadowingCovAPs = np.zeros((self.nAPs, self.nAPs))

    def computeCentroids(self):

        # --- Find the number of AP in a row
        self.nAPsPerRow = int(np.sqrt(self.nAPs))

        # --- Find the number of AP in a col
        self.nAPsPerCol = self.nAPsPerRow

        self.APsCentroids = [0] * self.nAPsPerRow
        for i in range(self.nAPsPerRow):
            self.APsCentroids[i] = [0] * self.nAPsPerCol

        # --- Compute the area
        self.area = self.D * self.D

        # --- Compute the area for each AP
        self.areaForAP = self.area / self.nAPs

        # --- Compute the length of the square edge
        lenght = np.sqrt(self.areaForAP)

        step = lenght

        # --- Collect all the centroids
        self.APsCoordinates = np.zeros((2, self.nAPs))
        self.APsCoordinates[0, 0] = lenght / 2
        self.APsCoordinates[1, 0] = lenght / 2
        idx = 0
        for r in range(self.nAPsPerRow - 1, -1, -1):
            for c in range(self.nAPsPerCol):
                self.APsCoordinates[0, idx] = (self.nAPsPerRow - 1 - r) * step + (lenght / 2)
                self.APsCoordinates[1, idx] = step * c + (lenght / 2)
                idx += 1






        # self.APsCentroids[self.nAPsPerRow - 1][0] = Centroid(lenght / 2, lenght / 2)
        # for r in range(self.nAPsPerRow - 1, -1, -1):
        #     for c in range(self.nAPsPerCol):
        #         self.APsCentroids[r][c] = Centroid((self.nAPsPerRow - 1 - r) * step + (lenght / 2), step * c + (lenght / 2))
        #
        # # --- Collect all the centroids
        # self.APsCoordinates = np.zeros((2, self.nAPs))
        #
        # idx = 0
        # for c in range(int(np.sqrt(self.nAPs))):
        #     for r in range(int(np.sqrt(self.nAPs))):
        #         self.APsCoordinates[0, idx] = self.APsCentroids[c][r].x
        #         self.APsCoordinates[1, idx] = self.APsCentroids[c][r].y
        #         idx += 1

    def plotCell(self):

        plt.scatter(self.APsCoordinates[0, :], self.APsCoordinates[1, :])
        
        
        
        if True or self.UEsCoordinates[0,0] != 0:
            plt.scatter(self.UEsCoordinates[0, :], self.UEsCoordinates[1, :])
            plt.legend(['APs', 'UEs'], loc='upper right')
            for i, txt in enumerate(np.arange(self.nUEs)):
                plt.annotate(txt, (self.UEsCoordinates[0, i], self.UEsCoordinates[1, i]))
        else:
            plt.legend(['APs'], loc='upper right')

        for i, txt in enumerate(np.arange(self.nAPs)):
            plt.annotate(txt, (self.APsCoordinates[0, i], self.APsCoordinates[1, i]))


        
        plt.grid()
        plt.xlim([0, self.D])
        plt.ylim([0, self.D])
        plt.xlabel('x-Axis')
        plt.ylabel('y-Axis')
        plt.show()


    def generateUsers(self):
        self.UEsCoordinates = np.random.rand(2, self.nUEs) * self.D

        self.computeLargeScaleCoeff()

    def computeLargeScaleCoeff(self):

        # === From APs to UEs
        for ap in range(self.nAPs):

            pointA = np.append(self.APsCoordinates[:,ap], self.ha)

            for ue in range(self.nUEs):
                pointB = np.append(self.UEsCoordinates[:,ue], self.hu)

                # --- Compute the Distance
                self.distAP2UE[ue, ap] = distance(pointA, pointB)

                # --- Compute the Path Loss (large scale coefficient)
                self.pathLoss[ue, ap] = np.sqrt(10 ** (self.computePathLossdB(self.distAP2UE[ue, ap])/10))

        self.contructShadowingCov()



    def contructShadowingCov(self):
        # === From UEs to UEs
        for ueR in range(self.nUEs):
            pointA = np.append(self.UEsCoordinates[:, ueR], self.hu)

            for ueC in range(ueR, self.nUEs, 1):

                pointB = np.append(self.UEsCoordinates[:, ueC], self.hu)

                if ueC != ueR:
                    # Compute distance
                    self.distUE2UE[ueR, ueC] = distance(pointA, pointB)

                    # Compute joint Expectation

                    self.shadowingCovUEs[ueR, ueC] = 2 ** (-self.distUE2UE[ueR, ueC]/self.dDecorr)
                else:
                    self.shadowingCovUEs[ueR, ueC] = 0.5

        temp = self.shadowingCovUEs + self.shadowingCovUEs.T
        self.shadowingCovUEs = temp

        # Find Eigenvalue decomposition of the covariance matrix
        S, U = np.linalg.eig(self.shadowingCovUEs)
        self.WUEs = np.dot(U, np.sqrt(np.diag(S)))


        # === From APs to APs
        for apR in range(self.nAPs):
            pointA = np.append(self.APsCoordinates[:, apR], self.ha)

            for apC in range(apR, self.nAPs, 1):

                pointB = np.append(self.APsCoordinates[:, apC], self.ha)

                if apC != apR:
                    # Compute distance
                    self.distAP2AP[apR, apC] = distance(pointA, pointB)

                    # Compute joint Expectation

                    self.shadowingCovAPs[apR, apC] = 2 ** (-self.distAP2AP[apR, apC]/self.dDecorr)
                else:
                    self.shadowingCovAPs[apR, apC] = 0.5

        temp = self.shadowingCovAPs + self.shadowingCovAPs.T
        self.shadowingCovAPs = temp

        S, U = np.linalg.eig(self.shadowingCovAPs)
        self.WAPs = np.dot(U,  np.sqrt(np.diag(S)))




    def computePathLossdB(self, distance):
        if distance > self.d1:
            pathLossTemp = 35 * np.log10(distance)
        elif distance > self.d0 and distance <= self.d1:
            pathLossTemp = 15 * np.log10(self.d1) + 20 * np.log10(distance)
        else:
            pathLossTemp = 15 * np.log10(self.d1) + 20 * np.log10(self.d0)

        # pathLossTemp = 15 * np.log10(self.d1) + 20 * np.log10(self.d0)

        return -self.L -pathLossTemp #+ np.random.normal(0,np.sqrt(8))

    def generateChannel(self):

        # --- Small scale
        self.h = np.sqrt(0.5) * (np.random.normal(0,1, (self.nUEs, self.nAPs * self.nAnts)) + 1j*np.random.normal(0,1, (self.nUEs, self.nAPs * self.nAnts)))

        # --- Large Scale
        # Shadowing
        shandowing = computeShadowing(self)


        self.large = self.pathLoss * np.sqrt(10 ** (shandowing/10))

        largeTemp = np.matlib.repmat(self.large, 1, self.nAnts)
        largeTemp = np.reshape(largeTemp, (self.nUEs, self.nAnts, self.nAPs))
        self.large = np.reshape(largeTemp.T, (self.nAnts * self.nAPs, self.nUEs)).T



        # return self.large
        self.H = self.h * self.large
        return self.H

def computeShadowing(self):
    # Generate a's
    z = np.random.normal(loc=0.0, scale=1, size=self.nUEs)
    a = np.dot(self.WUEs, z)

    # Generate b's
    z = np.random.normal(loc=0.0, scale=1, size=self.nAPs)
    b = np.dot(self.WAPs, z)

    Z = np.zeros((self.nUEs, self.nAPs))
    for ap in range(self.nAPs):
        Z[:, ap] = np.sqrt(self.delta) * a + np.sqrt(1-self.delta) * b[ap]

    return Z
def distance(pointA, pointB):
    return np.sqrt(np.sum((pointA - pointB)**2))

def computeL(fc, ha, hu):
    fc = fc / (1000 ** 2)
    return 46.3 + 33.9 * np.log10(fc) - 13.82 * np.log10(ha) - (1.11 * np.log10(fc) - 0.7)*hu + 1.56 * np.log10(fc) - 0.8
    # return -35.7
