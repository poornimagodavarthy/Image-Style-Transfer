import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image

#load and preprocess content image
content_path = "/Volumes/T7/StyleTransferProject/art.jpg"
content_img = image.load_img(content_path, target_size=(224, 224))
content_array = image.img_to_array(content_img)
content_array = image.img_to_array(content_img)
content_array = tf.expand_dims(content_array, axis=0)
content_input = preprocess_input(content_array)

# load and preprocess art style image
style_path = "/Volumes/T7/StyleTransferProject/photo.jpg"
style_img = image.load_img(style_path, target_size=(224, 224))
style_array = image.img_to_array(style_img)
style_array = tf.expand_dims(style_array, axis=0) #batch dimension
style_input = preprocess_input(style_array)

# normalize images after preprocessing.
content_input = content_input / 255.0
style_input = style_input / 255.0

#load VGG model, excluding fully connected layers (typically used for final classification task in original model)
vgg = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg.summary()

#define content and style layers
content_layer = vgg.get_layer('block5_conv2').output
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
style_layers = [vgg.get_layer(layer).output for layer in style_layers]


#create a feature extraction model
feature_extraction_model = tf.keras.Model(inputs=vgg.input, outputs=[content_layer, *style_layers])

#get feature maps
    #pass content and style images thru the model to get the feature maps
# Calculate content features for content image
content_features = feature_extraction_model(content_input)[0]

# Generate initial image for transfer and initialize with the content image
initial_generated_image = tf.Variable(content_input, dtype=tf.float32)

#np.random.uniform(-0.5, 0.5, size=(1, 224, 224, 3)) # add a batch dimension
generated_image = tf.Variable(content_img, dtype=tf.float32)

#generate an initial image for transfer and initialize with the content image
#generated_image = tf.Variable(tf.random.uniform(content_input.shape, -1, 1), dtype=tf.float32)

# Calculate content features for generated image
generated_content_features = feature_extraction_model(generated_image)[0]
print("generated content features shape: ", generated_content_features.shape)
#content and style weights
content_weight = 1.0
style_weight = 1.0

# Resize the generated image to match the dimensions of the content image
generated_image_resized = tf.image.resize(generated_image, (224, 224))
generated_image_resized = preprocess_input(generated_image_resized)
# Calculate content features for the resized generated image
generated_content_features_resized = feature_extraction_model(generated_image_resized)[0]

# Convert the generated_content_features_resized tensor to a NumPy array
generated_content_features_resized_np = generated_content_features_resized.numpy()

# Display or print the shape of the generated_content_features_resized
print("Generated content features resized shape:", generated_content_features_resized_np.shape)

#define content loss: diff of output vs content image, typically MSE
    # the goal is to minimize the loss in the optimization process
content_loss = content_weight * tf.reduce_mean(tf.square(content_features-generated_content_features_resized))

#define style loss
style_loss = 0

# Calculate style features for style image
style_features = feature_extraction_model(style_input)[1:]

# calculate gram matrices
    # captures the style info from the art
    # reshape feature maps & compute gram matrix
gram_matrices = [tf.linalg.einsum('bijc, bijd->bcd', style_feature, style_feature) / tf.cast(style_feature.shape[1]*style_feature.shape[2], tf.float32)
                 for style_feature in style_features]

# [1:]extracts the style feature maps from the list, excluding the content feature maps.
    # feature_extraction_model(generated_image) -> calculates the output of the feature extraction model when given the generated image as input
generated_style_features = feature_extraction_model(generated_image)[1:]

# calculates the style loss by comparing the gram matrices of the target style and generated image
    # iteratively finds the mean squared error
for target_gram, generated_gram in zip(gram_matrices, [tf.linalg.einsum('bijc, bijd->bcd', generated_style_feature, generated_style_feature) / tf.cast(generated_style_feature.shape[1]*generated_style_feature.shape[2], tf.float32) for generated_style_feature in generated_style_features]):
    style_loss += tf.reduce_mean(tf.square(target_gram - generated_gram), axis=[1, 2])

# combine style and content losses
    # multiplying style weight here is a design choice that allows for better control and balance between the content and style for the output
    # better hyperparameter tuning
total_loss = content_loss + style_weight * style_loss


# create an optimization loop: iteratively update generated image to minimize the loss using gradient descent
    # use tf.GradientTape to calculate gradients and update the image
optimizer = tf.compat.v1.keras.optimizers.Adam(learning_rate=0.001)  # Use the legacy Adam optimizer

#optimization loop
num_iterations = 1
for i in range(num_iterations):
    with tf.GradientTape() as tape:
        tape.watch(generated_image)
        # Calculate the style loss using the generated image features
        generated_style_features = feature_extraction_model(generated_image)[1:]
        style_loss = 0
        for target_gram, generated_gram in zip(gram_matrices, [tf.linalg.einsum('bijc, bijd->bcd', generated_style_feature, generated_style_feature) / tf.cast(generated_style_feature.shape[1]*generated_style_feature.shape[2], tf.float32) for generated_style_feature in generated_style_features]):
            style_loss += tf.reduce_mean(tf.square(target_gram - generated_gram), axis=[1, 2])

        # Calculate the total loss
        total_loss = content_loss + style_weight * style_loss
    
    # Calculate gradients of the total loss for the generated image
    gradients = tape.gradient(total_loss, generated_image)

    # Update the generated image with the optimizer
    optimizer.apply_gradients([(gradients, generated_image)])

    # Clip the pixel values to be in the valid range
    generated_image.assign(tf.clip_by_value(generated_image, 0.0, 255.0))

# Convert the generated image to a numpy array for visualization
generated_image_np = np.array(generated_image)

# Remove the batch dimension if it exists (assuming it's the first dimension)
if generated_image_np.shape[0] == 1:
    generated_image_np = generated_image_np[0]

# Display the generated image
generated_image_np = generated_image_np.astype('uint8')
plt.imshow(generated_image_np)
plt.show()
