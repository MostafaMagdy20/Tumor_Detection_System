import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import os


class ExtractFeatures:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def features(self, image):
        kp, descriptors = self.sift.detectAndCompute(image, None)
        return [kp, descriptors]


class Clustering:
    def __init__(self, clusters=20):
        self.clusters = clusters
        self.kmeans = KMeans(n_clusters=clusters)
        self.kmeans_res = None
        self.all_descriptors = None
        self.mega_histogram = None
        self.clf = SVC()  # classifier

    def cluster(self):

        self.kmeans_res = self.kmeans.fit_predict(self.all_descriptors)

    def developVocabulary(self, noImages, descriptorsList):  # Fill Clusters with Descriptors data

        self.mega_histogram = np.array([np.zeros(self.clusters) for i in range(noImages)])
        count = 0
        for i in range(noImages):
            size = len(descriptorsList[i])
            for j in range(size):
                idx = self.kmeans_res[count + j]
                self.mega_histogram[i][idx] += 1
            count += size
        print("Vocabulary Histogram Generated")

    def plotHistogram(self, histogram = None):
        if histogram is None:
            histogram = self.mega_histogram

        x_scalar = np.arange(self.clusters)
        y_scalar = np.array([abs(np.sum(histogram[:, h], dtype=np.int32)) for h in range(self.clusters)])

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Mega Histogram")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()

    def train(self, train_labels):
        """
        uses sklearn.svm.SVC classifier (SVM)
        """
        print("Training SVM")
        print(self.clf)
        print("Train labels", train_labels)
        self.clf.fit(self.mega_histogram, train_labels)
        print("************************************")
        print("Training Completed")
        print("************************************")



    def normalize(self, std=None):  # Normalization

        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)  # calc standard division
            self.mega_histogram = self.scale.transform(self.mega_histogram)  # transform
        else:
            self.mega_histogram = std.transform(self.mega_histogram)

    def makeDescriptorStack(self, listOfDisc):  # Append all descriptors in one vector
        vStack = np.array(listOfDisc[0])
        for remaining in listOfDisc[1:]:
            vStack = np.vstack((vStack, remaining))
        self.all_descriptors = vStack.copy()
        return


class ImageReader:
    def __init__(self):
        pass

    def getFiles(self, path):
        imageList = {}
        count = 0
        for each in os.listdir(path):
            print("-------------------------------------------------------")
            print("Reading image category --->", each)
            print("-------------------------------------------------------")
            imageList[each] = []
            for imagefile in os.listdir(path + '/' + each):
                print("Reading file", imagefile)
                im = cv2.imread(path + '/' + each + '/' + imagefile, 0)
                imageList[each].append(im)
                count += 1

        return [imageList, count]
