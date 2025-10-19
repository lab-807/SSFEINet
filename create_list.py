import os
path = 'data/Train/HSI'
imgs = os.listdir(os.path.join(path))
imgs.sort()
num = len(imgs)
fp = open(os.path.join('data/Train/Train.txt'), 'a')
fp.truncate(0)
for im in range(0,num):
    img_name = imgs[im]
    print(img_name)
    fp = open(os.path.join('data/Train/Train.txt'), 'a')
    fp.write(img_name)
    fp.write('\n')
    fp.close()



path = 'data/Test/HSI'
imgs = os.listdir(os.path.join(path))
imgs.sort()
num = len(imgs)
fp = open(os.path.join('data/Test/Test.txt'), 'a')
fp.truncate(0)
for im in range(0,num):
    img_name = imgs[im]
    print(img_name)
    fp = open(os.path.join('data/Test/Test.txt'), 'a')
    fp.write(img_name)
    fp.write('\n')
    fp.close()