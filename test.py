
import numpy as np
np.random.seed(123)
from glob import glob
from train import *
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session()


def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


def read_data(x, y):
    """ Read the image and mask from the given path. """
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    mask = cv2.imread(y, cv2.IMREAD_COLOR)
    return image, mask


def read_params():
    """ Reading the parameters from the JSON file."""
    with open("params.json", "r") as f:
        data = f.read()
        params = json.loads(data)
        return params


def load_data(path):
    """ Loading the data from the given path. """
    images_path = os.path.join(path, "image/*")
    masks_path  = os.path.join(path, "mask/*")

    images = glob(images_path)
    masks  = glob(masks_path)

    return images, masks


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def get_image(image_path, image_size_wight, image_size_height,gray=False):
    # load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
       
    if gray==True:
        img = img.convert('L')
    # center crop
    img_center_crop = img
    # resize
    img_resized = img
    edge = cv2.Canny(np.asarray(np.uint8(img_resized)),10,1000)
    
    flag = False
    # convert to numpy and normalize
    img_array = np.asarray(img_resized).astype(np.float32)/255.0
    edge = np.asarray(edge).astype(np.float32)/255.0
    #print(img_array)
    if gray==True:
        img_array=(img_array >=0.5).astype(int)
    img.close()
    return img_array,edge


test_img_list = glob("data/kdsb/test/images/*.jpg")
test_mask_list = glob("data/kdsb/test/masks/*.jpg")
print(test_img_list)
print(test_mask_list)
G = msrf()
G.load_weights('kdsb_ws.h5')
G.summary()
optimizer = get_optimizer()
G.compile(optimizer=optimizer, loss={'x':seg_loss, 'edge_out':'binary_crossentropy','pred4':seg_loss,'pred2':seg_loss},loss_weights={'x':1.,'edge_out':1.,'pred4':1. , 'pred2':1.})

X_tot_test = [get_image(sample_file,256,256) for sample_file in test_img_list]
X_test,edge_x_test = [],[]
for i in range(0,len(test_img_list)):
    X_test.append(X_tot_test[i][0])
    edge_x_test.append(X_tot_test[i][1])
X_test = np.array(X_test).astype(np.float32)
edge_x_test = np.array(edge_x_test).astype(np.float32)
print(edge_x_test.shape)
edge_x_test  =  np.expand_dims(edge_x_test,axis=3)
Y_tot_test = [get_image(sample_file,256,256,gray=True) for sample_file in test_mask_list]
Y_test,edge_y_test = [],[]
for i in range(0,len(test_img_list)):
    Y_test.append(Y_tot_test[i][0])
Y_test = np.array(Y_test).astype(np.float32)
           
Y_test  =  np.expand_dims(Y_test,axis=3)


y_pred,_,_,_ = G.predict([X_test,edge_x_test],batch_size=5)
y_pred = (y_pred >=0.5).astype(int)
res = mean_dice_coef(Y_test,y_pred)
print("dice coef on test set",res)


def compute_iou(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_true * y_pred).sum()

    #intersection = np.sum(intersection)   
    union = y_true.sum() + y_pred.sum() - intersection
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (intersection + 1e-15) / (union + 1e-15),tp/(tp+fp),tp/(tp+fn)


res = compute_iou(y_pred,Y_test)
print('iou on test set is ', res[0], " precision is ", res[1], " recall is ", res[2])