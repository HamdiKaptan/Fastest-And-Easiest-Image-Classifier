
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.vision import *
from fastai.metrics import error_rate
import os



os.mkdir("/kaggle/yourdatasetname")
kaynak = "/kaggle/input/dataname/dataname/"

hedef = "/kaggle/yourdatasetname/"
files = os.listdir(kaynak)
files.sort()
dosya_sayisi = 0
for f in files:
    dosya_sayisi += 1
    k = kaynak+f
    h = hedef+f
    shutil.copy(k,h)
 
print("%d adet dosya kopyalandı" %dosya_sayisi)
bs = 16


Datasetimizin yolunu değşkene atıyoruz.

path = pathlib.Path("/kaggle/yourdatasetname/")
path_img = path


fnames = get_image_files(path_img)
fnames[:5]


np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))

print(data.classes)
len(data.classes),data.c

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=299, bs=bs//2).normalize(imagenet_stats)
                                   
learn = cnn_learner(data, models.resnet50, metrics=error_rate)


learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(50)

interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=0)


img = learn.data.train_ds[1][0]
img

pred_class,pred_idx,outputs  = learn.predict(img)

pred_class

learn.export()
