import cv2
import numpy as np
from BOW_Helpers import *
from matplotlib import pyplot as plt


class BOW:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.imFeatures = ExtractFeatures()
        self.bowHelper = Clustering(no_clusters)
        self.fileReader = ImageReader()
        self.images = None
        self.trainImageCount = 0
        self.trainLabels = np.array([])
        self.nameDict = {}
        self.descriptorList = []

    def trainModel(self):

        self.images, self.trainImageCount = self.fileReader.getFiles(self.train_path)

        # extract SIFT Features from each image
        labelCount = 0
        for word, imgList in self.images.items():
            self.nameDict[str(labelCount)] = word
            print("Computing Features for --->", word)
            for im in imgList:
                self.trainLabels = np.append(self.trainLabels, labelCount)
                kp, des = self.imFeatures.features(im)
                self.descriptorList.append(des)

            labelCount += 1

        # perform clustering
        self.bowHelper.makeDescriptorStack(self.descriptorList)
        self.bowHelper.cluster()
        self.bowHelper.developVocabulary(noImages=self.trainImageCount, descriptorsList=self.descriptorList)
        # show trained histogram
        self.bowHelper.plotHistogram()
        self.bowHelper.normalize()
        self.bowHelper.train(self.trainLabels)

    def testAnalysis(self, testImg):

        kp, des = self.imFeatures.features(testImg)
        #print(des.shape)

        histo = np.array( [[ 0 for i in range(self.no_clusters)]])

        test_ret = self.bowHelper.kmeans.predict(des)

        for each in test_ret:
            histo[0][each] += 1

        # Scale the features
        vocab = self.bowHelper.scale.transform(histo)
        # predict the class of the image
        label = self.bowHelper.clf.predict(vocab)

        return label


    def testModel(self):

        self.testImages, self.testImageCount = self.fileReader.getFiles(self.test_path)

        predictions = []
        accuracy = 0

        for type, imglist in self.testImages.items():
            print("#############################")
            print ("Processing" ,type)
            for img in imglist:
               # print (img.shape)
                cl = self.testAnalysis(img)
                #print (cl)
                predictions.append({
                    'image' : img ,'class Label' : cl ,'Class Name' : self.nameDict[str(int(cl[0]))]
                    })

                if(self.nameDict[str(int(cl[0]))] == type):
                    accuracy = accuracy + 1
        print("************************************")
        print("Testing Completed")
        print("************************************")
        print("Test Accuracy = " + str((accuracy / self.testImageCount) * 100))
        print("************************************")

        for i in predictions:
            plt.imshow(i['image'] , cmap='gray')
            plt.title(i['Class Name'])
            plt.show()


if __name__ == '__main__':

    bow = BOW(no_clusters = 200)

    # set training paths
    bow.train_path = (r"C:\Users\mosta\PycharmProjects\Vision_Project\Dataset\Train")
    # set testing paths
    bow.test_path = (r"C:\Users\mosta\PycharmProjects\Vision_Project\Dataset\Test")
    # train the model
    bow.trainModel()
    # test model
    bow.testModel()

