import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

root_path='/mnt/hgfs/Book/kaggle/'
''' root_path contains: driver_img_list.csv/,
                        sample_submission.csv/
                        imgs/'''
width = 640
height = 480

def get_pic(path):
    '''return array (height, width, 3)'''
    return mpimg.imread(path)

def show_pic(pic):
    '''pic - vector (height, width, 3)'''
    plt.imshow(pic)
    plt.show()

def get_train_paths():
    '''return [(person, class(int), filename)]'''
    driver_path = root_path + 'driver_imgs_list.csv/driver_imgs_list.csv'
    f = open(driver_path)
    f.readline()
    res = []
    for line in f.readlines():
        [person, label, fp] = [s.strip() for s in line.split(',')]
        res.append((person, int(label[-1]), fp))
    return res
    
def get_train_pic(tup):
    '''
    tup - [(person, class(int), filename)]
    return pic array (width, height, 3)'''
    [person, label, fn] = tup
    path = root_path + 'imgs/train/c' + str(label) + '/' + fn
    return get_pic(path)
    
def split_datas(tup, cv_rate=0.2, test_rate=0.2):
    '''return [train_data, cv_data, test_data]'''
    np.random.shuffle(tup)
    l = len(tup)
    cv_data = tup[:int(l*cv_rate)]
    test_data = tup[int(l*cv_rate):int(l*(cv_rate+test_rate))]
    train_data = tup[int(l*(cv_rate+test_rate)):]
    return (train_data, cv_data, test_data)
    
