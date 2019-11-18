import os
from os.path import join, dirname
import numpy as np
import pandas as pd
import imageio
from skimage import color
import cv2
from tf_unet import unet

class MyGenerator(object):

    def __init__(self):
        base = dirname(dirname(__file__))
        df = pd.read_csv(join(base, "train.csv"))
        df["filename"] = df["Image_Label"].apply(lambda x : x.split("_")[0])
        df["labelname"] = df["Image_Label"].apply(lambda x : x.split("_")[1])
        self.df = df
        self.shape = (1400, 2100)
        self.nx = 350
        self.ny = 525
        self.imagepath = join(base, "train_images")
        self.channels = 1
        self.n_class = 2
        self.a_max = np.inf
        self.a_min = -np.inf
        self.kwargs = {'cnt': 20}
        self.resize = True
#        self.resize = False

    def rle2mask(self, encoded_label):
        """
        Source
        ------
        https://www.kaggle.com/saneryee/understanding-clouds-keras-unet
        """
        width, height = self.shape

        mask= np.zeros( width*height ).astype(np.uint8)

        if encoded_label is np.nan:
            return mask.reshape(height, width).T

        array = np.asarray(encoded_label.split(' ')).astype(np.int)
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            mask[int(start):int(start+lengths[index])] = 1
            current_position += lengths[index]

        return mask.reshape(height, width).T

    def _resize(self, image):
        retval = cv2.resize(image, (self.ny, self.nx))
        return retval


    def _imread(self, filename, gray=True):
        """ wrapper for imageio.imread """
        retval =  imageio.imread(join(self.imagepath, filename))
        if gray:
            retval= color.rgb2gray(retval)

        return retval

    def __call__(self, nof):
        """ generate training data and labels in the format needed by unet

        Returns
        -------
        x : np.array of shape (nofc, xdim, ydim, 1 or 3)
        y : np.array of shaoe (nofc, xdim, ydim, 2)
        """
        idxs = np.random.randint(0, len(self.df), nof)

        if self.resize:
            lcs = lambda f, n : np.stack([self._resize(f(self.df.iloc[idx][n]))
                                            for idx in idxs])
        else:
            lcs = lambda f, n : np.stack([f(self.df.iloc[idx][n])
                                          for idx in idxs])


        images = lcs(self._imread, "filename")
        labels = lcs(self.rle2mask, "EncodedPixels")
        inv = 1 - labels

        prep = lambda al : np.moveaxis(np.stack(al), 0, 3).astype(np.float64)
        return prep([images]), prep([inv, labels])

class OneClassGenerator(MyGenerator):
    """ Subclass of MyGenerator used for specific label classes

    Options: fish, Flower, Gravel, Sugar
    """

    def __init__(self, labelclass):
        assert(labelclass in ["Fish", "Flower", "Gravel", "Sugar"])
        super().__init__()
        self.labelclass = labelclass
        self.df = self.df.dropna().query("labelname == '%s'" % (labelclass))


def train(gen):
    net = unet.Unet(channels=gen.channels, n_class=gen.n_class, layers=5,
                    features_root=16)
    trainer = unet.Trainer(net, optimizer="momentum",
                           opt_kwargs=dict(momentum=0.2))
    trainer.train(gen, "./unet_trained/%s" % (gen.labelclass),
                  training_iters=32, epochs=100, display_step=2)

class MyPredictor(MyGenerator):

    def __init__(self):
        self.net = unet.Unet(channels=1, n_class=2, layers=5,
                             features_root=16)
        base = dirname(dirname(__file__))
        self.imagepath = join(base, "testimages")
        self.models = ["Fish", "Flower", "Gravel", "Sugar"]
        self.testimages = os.listdir(self.imagepath)
        self.nx = 350
        self.ny = 525
        self.mt = {"Fish": .5, "Gravel": .38263,
                   "Flower": .4999, "Sugar": .39}
        self.cropx = 196
        self.cropy = 192

    @staticmethod
    def _modelpath(model):
        return join(dirname(__file__), "unet_trained", model, "model.ckpt")

    def predict(self, data, model="Fish"):
        retval = self.net.predict(self._modelpath(model), data)
        return self._postresult(retval, model)

    @staticmethod
    def _fixdim(arr):
        """ (x, y) -> (1, x, y, 1)"""
        return np.moveaxis(np.array([[arr]]), 0, 3)

    def _postresult(self, arr, model, resize=False):
        retval = (arr[0, ..., 1] > self.mt[model]).astype(np.uint8)
        if resize:
            retval = cv2.resize(retval, (self.ny, self.nx))
        return retval

    def _crop(self, arr):
        return arr[int(self.cropx/2): -int(self.cropx/2),
                   int(self.cropy/2): -int(self.cropy/2)]

    def prettypredict(self, num, model, strength=75):
        """ """
        img = self._imread(self.testimages[num], gray=False)
        gray = self._fixdim(self._imread(self.testimages[num], gray=True))
        mask = self.predict(gray, model)

        #crop and resize
        img = self._crop(img)
        img = cv2.resize(img, mask.shape[::-1])

        retval = (img * 0.6).astype(np.uint8)
        retval[:, :, 1] += strength * mask # change here for different color
        return retval

