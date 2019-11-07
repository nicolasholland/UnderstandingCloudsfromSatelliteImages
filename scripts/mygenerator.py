from os.path import join, dirname
import numpy as np
import pandas as pd
import imageio
from skimage import color
from skimage.transform import resize
from tf_unet import unet

class MyGenerator(object):

    def __init__(self):
        base = dirname(dirname(__file__))
        df = pd.read_csv(join(base, "train.csv"))
        df["filename"] = df["Image_Label"].apply(lambda x : x.split("_")[0])
        df["labelname"] = df["Image_Label"].apply(lambda x : x.split("_")[1])
        self.df = df
        self.nx = int(1400 / 4)
        self.ny = int(2100 / 4)
        self.imagepath = join(base, "train_images")
        self.channels = 1
        self.n_class = 2
        self.a_max = np.inf
        self.a_min = -np.inf
        self.kwargs = {'cnt': 20}
        self.resize = True

    def rle2mask(self, encoded_label):
        """
        Source
        ------
        https://www.kaggle.com/saneryee/understanding-clouds-keras-unet
        """
        width, height = self.nx, self.ny

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
        retval = resize(image, (self.nx, self.ny), anti_aliasing=True)
        return retval


    def _imread(self, filename):
        """ wrapper for imageio.imread """
        retval =  imageio.imread(join(self.imagepath, filename))
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
            lcs = lambda f, n : np.stack([f(self.df.iloc[idx][n]) for idx in idxs])


        images = lcs(self._imread, "filename")
        labels = lcs(self.rle2mask, "EncodedPixels")

        images = np.moveaxis(np.stack([images]), 0, 3)
        inv = 1 - labels
        labels = np.moveaxis(np.stack([labels, inv]), 0, 3)

        return images.astype(np.float64), labels.astype(np.float64)


mg = MyGenerator()
net = unet.Unet(channels=mg.channels, n_class=mg.n_class, layers=3, features_root=16)
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(mg, "./unet_trained", training_iters=32, epochs=10, display_step=2)

