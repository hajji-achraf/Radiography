import tensorflow as tf
import os
import cv2
from sklearn.metrics import precision_recall_curve
import imghdr
import ssl
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.metrics import precision_recall_curve

# Configurer l'utilisation du GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Définir le répertoire des données et les extensions d'image
data_dir = 'C:\\Users\\LENOVO\\Downloads\\data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

# Vérification et nettoyage des images
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in extlist: {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

# Charger le dataset
data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Afficher les informations du batch
print(batch[0].shape)
print(batch[1])

# Obtenir les noms des classes
class_names = data.class_names

# Afficher les images avec les étiquettes
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    label = batch[1][idx]
    class_name = class_names[label]
    ax[idx].title.set_text(f"Classe: {class_name}, Indice: {label}")
plt.show()

# Prétraiter les données
data = data.map(lambda x, y: (x / 255.0, y))

# Créer un nouvel itérateur
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

# Afficher la longueur du dataset
print(len(data))
print(batch[0].min())
print(batch[0].max())

# Diviser le dataset en ensembles d'entraînement, validation, test
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2) + 1
test_size = int(len(data) * 0.1)
print(train_size, val_size, test_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Construction du modèle
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilation du modèle
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

# Entraînement du modèle
logdir = 'C:\\Users\\LENOVO\\Downloads\\logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# Tracé des courbes de perte et de précision
fig = plt.figure()
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='red', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Évaluation du modèle
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

y_true = []
y_scores = []

for batch in test.as_numpy_iterator():
    x, y = batch
    y_true.extend(y)
    y_scores.extend(model.predict(x).flatten())
    pre.update_state(y, model.predict(x))
    re.update_state(y, model.predict(x))
    acc.update_state(y, model.predict(x))

print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

# Calculer la courbe de précision-rappel
y_true = np.array(y_true)
y_scores = np.array(y_scores)
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# Tracer la courbe de précision-rappel
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Rappel')
plt.ylabel('Précision')
plt.title('Courbe de Précision-Rappel')
plt.grid(True)
plt.show()

# Test sur une nouvelle image
img = cv2.imread('C:\\Users\\LENOVO\\Downloads\\testimage.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

resize = tf.image.resize(img, (256, 256))
yhat = model.predict(np.expand_dims((resize / 255.0), 0))

if yhat > 0.5:
    print('Normal')
else:
    print('Anormale')

# Save the model
model.save("models\\model_radiography.h5")
