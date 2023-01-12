from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob

IMAGE_SIZE = [224, 224]

train_path = '/content/drive/MyDrive/btprojectfolders/train'
valid_path = '/content/drive/MyDrive/btprojectfolders/test'

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in inception.layers:
    layer.trainable = False

folders = glob('/content/drive/MyDrive/btprojectfolders/train/*')

x = Flatten()(inception.output)

prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=inception.input, outputs=prediction)

model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/btprojectfolders/train',
                                                 target_size = (224, 224),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/btprojectfolders/test',
                                            target_size = (224, 224),
                                            batch_size = 64,
                                            class_mode = 'categorical')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

from tensorflow.keras.models import load_model

model.save('model_inception.h5')
model = load_model('model_inception.h5')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

import matplotlib.pyplot as plt

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_inception.h5')

y_pred = model.predict(test_set)

y_pred

import numpy as np
y_pred = np.argmax(y_pred, axis=1)

y_pred

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

img=image.load_img('/content/drive/MyDrive/btprojectfolders/test/dress/0e7d1a99-f073-470b-bd61-e77062171de5_size624x818.jpg',target_size=(224,224))

x=image.img_to_array(img)
x

x.shape

print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

x=x/255
import numpy as np
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape

test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/btprojectfolders/test',
                                            target_size = (224, 224),
                                            batch_size = 64,
                                            class_mode = 'categorical')

print(test_set.classes)

import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = ['coat', 'dress', 'panths', 'pullover', 'shirt', 'shorts', 'skirt', 'suit', 'sweatshirt', 'tshirt']
print(classification_report(np.argmax(test_set,axis=1), y_pred,target_names=target_names))

def prepare_image(file):
  img_path = '/content/drive/MyDrive/btprojectfolders/test/skirt/'
  img = image.load_img(img_path + file, target_size=(224, 224))
  x = image.img_to_array(img)
  img_array_expanded_dims = np.expand_dims(x, axis=0)
  return tf.keras.applications.inception_v3.preprocess_input(img_array_expanded_dims)

Image(filename='/content/drive/MyDrive/btprojectfolders/test/skirt/0ccb090e-7591-4bde-b02a-254202c5fa2e_size624x818.jpg', width=300, height=200)

from keras.applications.inception_v3 import preprocess_input, decode_predictions
if __name__ == '__main__':
    

    img_path = '/content/drive/MyDrive/btprojectfolders/test/skirt/0ccb090e-7591-4bde-b02a-254202c5fa2e_size624x818.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
    

Image(filename='/content/drive/MyDrive/btprojectfolders/test/pullover/5002954684_332_01.jpg', width=300, height=200)

from keras.applications.inception_v3 import preprocess_input, decode_predictions
if __name__ == '__main__':
    

    img_path = '/content/drive/MyDrive/btprojectfolders/test/pullover/5002954684_332_01.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
