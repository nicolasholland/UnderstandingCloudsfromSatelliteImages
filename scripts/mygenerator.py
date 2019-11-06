from os.path import join, dirname
import numpy as np
import pandas as pd
import imageio


class MyGenerator(object):

    def __init__(self):
        base = dirname(dirname(__file__))
        df = pd.read_csv(join(base, "train.csv"))
        df["filename"] = df["Image_Label"].apply(lambda x : x.split("_")[0])
        df["labelname"] = df["Image_Label"].apply(lambda x : x.split("_")[1])
        self.df = df
        self.input_shape = (1400, 2100)
        self.imagepath = join(base, "train_images")

    def rle2mask(self, encoded_label):
        """
        Source
        ------
        https://www.kaggle.com/saneryee/understanding-clouds-keras-unet
        """
        width, height = self.input_shape

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

    def _imread(self, filename):
        """ wrapper for imageio.imread """
        return imageio.imread(join(self.imagepath, filename))

    def __call__(self, nof):
        """ generate training data and labels in the format needed by unet """
        idxs = np.random.randint(0, len(self.df), nof)

        lcs = lambda f, n : np.stack([f(self.df.iloc[idx][n]) for idx in idxs])

        images = lcs(self._imread, "filename")
        labels = lcs(self.rle2mask, "EncodedPixels")

        return images, labels


mg = MyGenerator()

