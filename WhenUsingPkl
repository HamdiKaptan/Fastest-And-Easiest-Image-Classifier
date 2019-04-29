import cv2
from fastai.vision import *


learn = load_learner(Path('E:\İndirmeler'), fname=Path('export.pkl'))

vidcap = cv2.VideoCapture('2_kisi_A.mp4')

while True:
    ret, frame = vidcap.read()
    pred_class = learn.predict(Image(pil2tensor(frame, np.float32).div_(255)))[0]
    print(pred_class)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows()
