#encoding=utf-8
import numpy as np
import cv2
from skimage import feature as ft
import matplotlib.pyplot as plt
# import torchvision.models as models
#import pretrainedmodels
#import pretrainedmodels.utils as utils
# import torch
import os, sys
import json
from sklearn.decomposition import PCA

#load_img = utils.LoadImage()

def getHOG(imfn):
    img = cv2.imread(imfn)
    img = cv2.resize(img, (64,64))
    features = ft.hog(img, orientations=6, pixels_per_cell=[8, 8], cells_per_block=[2, 2], visualize=True)
    return features[1]
    # plt.imshow(features[1], cmap=plt.cm.gray)
    # plt.show()


def getCanny(imfn):
    img = cv2.imread(imfn)
    im = cv2.resize(img, (64, 64))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    features = ft.canny(im)
    return features
    # plt.imshow(edges1, cmap=plt.cm.gray)
    # plt.show()

def getImageEmbedding(imfn):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    model_name = 'inceptionv3'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    tf_img = utils.TransformImage(model)
    input_img = load_img(imfn)
    input_tensor = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
    input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
    input = torch.autograd.Variable(input_tensor,
                                    requires_grad=False)
    output_features = model.features(input)
    emb = output_features[0].detach().numpy()
    return emb

imgfn = "3/030000_S29.png"

def getAllImage():
    sons = os.listdir()
    image_fn_list = []
    for son in sons:
        name = os.path.join("./", son)
        if os.path.isdir(name):
            for fn in os.listdir(name):
                tmp_fn = os.path.join(name, fn)
                if 'png' in tmp_fn:
                    image_fn_list.append(tmp_fn)
    return image_fn_list


def getEmb():
    image_fns = getAllImage()
    fw = open('emb.txt', 'w')
    for fn in image_fns:
        hog_features = getHOG(fn).flatten()
        canny = getCanny(fn).flatten()
        inception_emb = getImageEmbedding(fn).flatten()
        data = {
            'hog': hog_features.tolist(),
            'canny': canny.tolist(),
            'inception:': inception_emb.tolist()
        }
        fw.write(fn + "_" + json.dumps(data) + '\n')
        # res_dict[fn] = data
    fw.close()


def getPCA():
    lines = open('emb.txt').readlines()
    fn_list = []
    hog_list = []
    canny_list = []
    inception_list = []
    for line in lines:
        res = line.strip().split('_')
        fn = "_".join(res[:-1])
        # print(res[-1])
        data = json.loads(res[-1])
        fn_list.append(fn)
        hog_list.append(data['hog'])
        canny_list.append(data['canny'])
        inception_list.append(data['inception:'])

    hogs = np.array(hog_list, dtype=np.float32)
    pca_1 = PCA(n_components=64)
    hogs_pca = pca_1.fit_transform(hogs)
    print('===hog done.')

    canny = np.array(canny_list, dtype=np.float32)
    pca_2 = PCA(n_components=64)
    canny_pca = pca_2.fit_transform(canny)
    print('===canny done.')

    inception = np.array(inception_list, dtype=np.float32)
    pca_3 = PCA(n_components=1024)
    inception_pca = pca_3.fit_transform(inception)
    print('===inception done.')

    fw = open('pca.txt', 'w')
    for i in range(len(fn_list)):
        fn = fn_list[i]
        data = {
            'hog': hogs_pca[i].tolist(),
            'canny':canny_pca[i].tolist(),
            'inception': inception_pca[i].tolist()
        }
        fw.write(fn + '_' + json.dumps(data) + '\n')
    fw.close()

def cluster():
    f = open('pca.txt')
    fn_list = []
    emb_list = []
    for line in f:
        res = line.strip().split('_')
        fn = "_".join(res[:-1])
        data = json.loads(res[-1])
        emb_list.append(data['hog'] + data['canny'] + data['inception'])
        fn_list.append(fn)

    for n_clusters in [30,60,100]:
        emb = np.array(emb_list, dtype=np.float)
        from sklearn.cluster import KMeans
        clf = KMeans(n_clusters=n_clusters)
        y = clf.fit_predict(emb)
        fw = open('kmeans_'+ str(n_clusters) + ".txt", 'w')
        for i in range(len(fn_list)):
            fw.write(fn_list[i] + ':' + str(y[i]) + '\n')
            # print(y[i], fn_list[i])
        fw.close()

        distorsions = []
        # for n_clusters in [10, 30, 50, 70, 90]:
        #for n_clusters in [10, 30, 50, 70, 90]:
        #    clf = KMeans(n_clusters=n_clusters)
        #    clf.fit_predict(emb)
        #    distorsions.append(clf.inertia_)
        #fig = plt.figure(figsize=(15, 5))
        #plt.plot([10, 30, 50, 70, 90], distorsions)
        #plt.grid(True)
        #plt.title('Elbow curve')
        #plt.savefig("elbow_kmeans.jpg")

        from sklearn.cluster import AgglomerativeClustering
        y = AgglomerativeClustering(linkage='ward', n_clusters=n_clusters).fit_predict(emb)

        fw = open('agg_cluster_'+str(n_clusters) + ".txt", 'w')
        for i in range(len(fn_list)):
            fw.write(fn_list[i] + ':' + str(y[i]) + '\n')
            # print(y[i], fn_list[i])
        fw.close()
    # for i in range(len(fn_list)):
    #     print(y[i], fn_list[i])


def run():
    cluster()
    # getPCA()







run()
# getImageEmbedding(imgfn)
# getCanny(imgfn)
# getHOG(imgfn)

#

#
