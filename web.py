import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Charger le modèle
model = tf.keras.models.load_model("models\\model_radiography.h5")


# Fonction pour faire des prédictions sur une image
def predict(image):
    # Redimensionner l'image à la taille d'entrée du modèle
    image_resized = tf.image.resize(image, (256, 256))
    # Normaliser l'image
    image_normalized = image_resized / 255.0
    # Ajouter une dimension pour correspondre à la forme attendue (1, 256, 256, 3)
    image_batch = np.expand_dims(image_normalized, axis=0)
    # Faire la prédiction
    prediction = model.predict(image_batch)
    # Retourner la prédiction (0 ou 1)
    return prediction[0][0]


# Couleurs pour le texte
header_color = "blue"
subheader_color = "grey"
text_color = "black"
background_color = "#f0f0f0"

# Mise en page du site web
st.markdown(f"""
    <style>
    .reportview-container {{
        background-color: {background_color};
    }}
    .header {{
        color: {header_color};
        font-size: 36px;
        font-weight: bold;
    }}
    .subheader {{
        color: {subheader_color};
        font-size: 28px;
        font-weight: bold;
    }}
    .text {{
        color: {text_color};
        font-size: 20px;
    }}
    ul {{
        color: {text_color};
        font-size: 20px;
    }}
    </style>
    """, unsafe_allow_html=True)

# Onglets pour diviser la présentation et la prédiction
tab1, tab2 = st.tabs(["CNN Presentation", "Radiography Prediction"])

with tab1:
    # Présentation du CNN
    st.markdown("<div class='header'>What is a Convolutional Neural Network (CNN)?</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='text'>
        A Convolutional Neural Network (CNN), also known as ConvNet, is a specialized type of deep learning algorithm mainly designed for tasks that necessitate object recognition, including image classification, detection, and segmentation. CNNs are employed in a variety of practical scenarios, such as autonomous vehicles, security camera systems, and others.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='subheader'>The importance of CNNs</div>", unsafe_allow_html=True)
    st.image("images/image1.jpg")
    st.markdown(f"""
        <div class='text'>
        There are several reasons why CNNs are important in the modern world, as highlighted below:
        <ul>
            <li>CNNs are distinguished from classic machine learning algorithms such as SVMs and decision trees by their ability to autonomously extract features at a large scale, bypassing the need for manual feature engineering and thereby enhancing efficiency.</li>
            <li>The convolutional layers grant CNNs their translation-invariant characteristics, empowering them to identify and extract patterns and features from data irrespective of variations in position, orientation, scale, or translation.</li>
            <li>A variety of pre-trained CNN architectures, including VGG-16, ResNet50, Inceptionv3, and EfficientNet, have demonstrated top-tier performance. These models can be adapted to new tasks with relatively little data through a process known as fine-tuning.</li>
            <li>Beyond image classification tasks, CNNs are versatile and can be applied to a range of other domains, such as natural language processing, time series analysis, and speech recognition.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='subheader'>Inspiration Behind CNN and Parallels With The Human Visual System</div>",
                unsafe_allow_html=True)
    st.markdown(f"""
        <div class='text'>
        Convolutional neural networks were inspired by the layered architecture of the human visual cortex, and below are some key similarities and differences:
        <ul>
            <li><strong>Hierarchical architecture:</strong> Both CNNs and the visual cortex have a hierarchical structure, with simple features extracted in early layers and more complex features built up in deeper layers. This allows increasingly sophisticated representations of visual inputs.</li>
            <li><strong>Local connectivity:</strong> Neurons in the visual cortex only connect to a local region of the input, not the entire visual field. Similarly, the neurons in a CNN layer are only connected to a local region of the input volume through the convolution operation. This local connectivity enables efficiency.</li>
            <li><strong>Translation invariance:</strong> Visual cortex neurons can detect features regardless of their location in the visual field. Pooling layers in a CNN provide a degree of translation invariance by summarizing local features.</li>
            <li><strong>Multiple feature maps:</strong> At each stage of visual processing, there are many different feature maps extracted. CNNs mimic this through multiple filter maps in each convolution layer.</li>
            <li><strong>Non-linearity:</strong> Neurons in the visual cortex exhibit non-linear response properties. CNNs achieve non-linearity through activation functions like ReLU applied after each convolution.</li>
        </ul>
        CNNs mimic the human visual system but are simpler, lacking its complex feedback mechanisms and relying on supervised learning rather than unsupervised, driving advances in computer vision despite these differences.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='subheader'>Key Components of a CNN</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='text'>
        The convolutional neural network is made of four main parts.
        <ul>
            <li>Convolutional layers</li>
            <li>Rectified Linear Unit (ReLU for short)</li>
            <li>Pooling layers</li>
            <li>Fully connected layers</li>
        </ul>
        This section dives into the definition of each one of these components through the example of the following example of classification of a handwritten digit.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='subheader'>Convolution layers</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='text'>
        This is the first building block of a CNN. As the name suggests, the main mathematical task performed is called convolution, which is the application of a sliding window function to a matrix of pixels representing an image. The sliding function applied to the matrix is called kernel or filter, and both can be used interchangeably.
        <br><br>
        In the convolution layer, several filters of equal size are applied, and each filter is used to recognize a specific pattern from the image, such as the curving of the digits, the edges, the whole shape of the digits, and more.
        <br><br>
        Put simply, in the convolution layer, we use small grids (called filters or kernels) that move over the image. Each small grid is like a mini magnifying glass that looks for specific patterns in the photo, like lines, curves, or shapes. As it moves across the photo, it creates a new grid that highlights where it found these patterns.
        <br><br>
        For example, one filter might be good at finding straight lines, another might find curves, and so on. By using several different filters, the CNN can get a good idea of all the different patterns that make up the image.
        <br><br>
        Let’s consider this 32x32 grayscale image of a handwritten digit. The values in the matrix are given for illustration purposes.
        </div>
        """, unsafe_allow_html=True)

    st.image("images/image2.jpg")
    st.markdown(f"""
        <div class='text'>
        Also, let’s consider the kernel used for the convolution. It is a matrix with a dimension of 3x3. The weights of each element of the kernel is represented in the grid. Zero weights are represented in the black grids and ones in the white grid.
        <br><br>
        </div>
        """, unsafe_allow_html=True)

    st.image("images/image3.jpg")

    st.markdown("<div class='subheader'>Activation function</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='text'>
        A ReLU activation function is applied after each convolution operation. This function helps the network learn non-linear relationships between the features in the image, hence making the network more robust for identifying different patterns. It also helps to mitigate the vanishing gradient problems.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='subheader'>Pooling layer</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='text'>
        The goal of the pooling layer is to pull the most significant features from the convoluted matrix. This is done by applying some aggregation operations, which reduce the dimension of the feature map (convoluted matrix), hence reducing the memory used while training the network. Pooling is also relevant for mitigating overfitting.
        <br><br>
        The most common aggregation functions that can be applied are:
        <ul>
            <li>Max pooling, which is the maximum value of the feature map</li>
            <li>Sum pooling corresponds to the sum of all the values of the feature map</li>
            <li>Average pooling is the average of all the values.</li>
        </ul>
        Below is an illustration of each of the previous example:
        <br><br>
        Application of max pooling with a stride of 2 using 2x2 filter
        <br><br>
        Also, the dimension of the feature map becomes smaller as the pooling function is applied.
        <br><br>
        The last pooling layer flattens its feature map so that it can be processed by the fully connected layer.
        </div>
        """, unsafe_allow_html=True)

    st.image("images/image4.jpg")

    st.markdown("<div class='subheader'>Fully connected layers</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class='text'>
        These layers are in the last layer of the convolutional neural network, and their inputs correspond to the flattened one-dimensional matrix generated by the last pooling layer. ReLU activations functions are applied to them for non-linearity.
        <br><br>
        Finally, a softmax prediction layer is used to generate probability values for each of the possible output labels, and the final label predicted is the one with the highest probability score.
        </div>
        """, unsafe_allow_html=True)

with tab2:
    # Prédiction de radiographies
    st.markdown("<div class='header'>Radiography Disease Prediction</div>", unsafe_allow_html=True)


    st.markdown("<div class='subheader'>Upload Image</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a radiographic image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
            # Lire l'image via PIL
            image = Image.open(uploaded_file)
            # Afficher l'image téléchargée
            st.image(image, caption="Uploaded Image", use_column_width=True)


    st.markdown("<div class='subheader'>Prediction Result</div>", unsafe_allow_html=True)
    if uploaded_file is not None:
            # Convertir l'image en tableau numpy pour la passer à OpenCV
            image_np = np.array(image)

            # Afficher un bouton pour lancer la prédiction
            if st.button("Predict"):
                # Faire la prédiction
                prediction = predict(image_np)

                # Afficher le résultat
                if prediction > 0.5:
                    st.write("The model predicts: **Normal**")
                else:
                    st.write("The model predicts: **Anormal**")
