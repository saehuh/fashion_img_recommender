# Product Recommendation based on image similarity

## Goal: 
The goal of this project was to create a recommender system based on similarity of the product images from a catalogue and compare it to use of pretrained model(Resnet50) trained with much larger, complex and general purpose dataset(ImageNet).

## Background:

Having been a product manager for E-commerce platforms for various types of fashion brands, optimizing recommender system has been one of the biggest challenges. It involves extensive tagging or data entry in order to build clusters of similar or related products to effectively recommend what the customer may be looking for. However, in the fashion industry, the most important feature of an item is actually the presentation of the image. Customers look at the image and imagine how it would look on themselves before making the purchase decision. This project explores product recommendation system using the similarity of the images in the product catalog.

## Data Source
![img_example](/img/img_example.png)

Here I am using Kaggle's Fashion Product Images Dataset. The data is comprised of 5000 professionally shot product images and multiple label attributes decsribing the product such as master category, sub category, article type, base color, season, year, usage and product display name. For this project, I used article type as the label for training the models, which had 141 unique classes.

I built a utility function to get images inserted into image array where X is image data and Y is the label data. Initially, I reshaped the images into 28 by 28 pixels size. 

## Recommendation Systems

![RecSys_Arch](/img/recsys_arch.png)


Candidate generation: Usually a large corpus of data is available but it has a smaller number of candidates or categories.Generation and differentitation of these categories is the first step of building the recommendation system.

Scoring: Scoring is the model that creates subsets and ranks the categories to be displayed to the user. The customization of scoring of categories can be done using conditions or queries.

Re-ranking: The final step, re-ranking considers all available constraints and removes unwanted or low-scoring recommendations. Re-ranking is extremely important as it filters all unnecessary items.


## CNN

![CNN_Model](/img/cnn_model.png)


A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.

Here is the summary of my model:
![CNN_Model_summary](/img/cnn_modelsum.png)

Below is the evolution of loss and accuray over 150 epochs of training.
>> cnn_evolution
![CNN_Evolution](/img/cnn_evolution.png)


As a result, we got a model that classifies images from our dataset into 141 different unique classes with 80% accuracy.

![CNN_Classification](/img/cnn_classification.png)

In the recommender system, we use this model to classify an image and I created a list of similar products using the identified article type as the label data. Once an image is classified, the recommender system checks for other images that belong to the same class and ranks them by similarities caculated using either vector distance or cosine similarities.

Here is an example of recommended items for a handbag.
![Rec_results](/img/rec_results.png)


This recommender system worked fine as far as finding other items that belong to the same class and calculating the similarities from model. But, I realized the labels that were used in the training process of the model had its own limitations. 

Customers, especially in fashion, aren’t always looking to purchase things in the same category as what they are looking at the moment. There has to be a bridge that connects the user from one section of the product catalog to another to keep them interested. So I tried using pretrained model that wasn’t trained on our own dataset for recommender system. 


## Using pretrained Resnet50 model to get embeddings of images
![Resnet50](/img/resnet50.png)

I am using Resnet50 without training a model against specific label. I am just getting embeddings of our image data from Resnet50's complex structure as a kind of unsupervised learning, which reduces the size of the demension from 36,720 to 2,048. 

An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors. Embeddings make it easier to do machine learning on large inputs like sparse vectors representing words. Ideally, an embedding captures some of the semantics of the input by placing semantically similar inputs close together in the embedding space. An embedding can be learned and reused across models.

The secret sauce of deep neural networks is the rich feature engineering done in the layer before the classifier. You can think of this learned data representation as an embedding. Essentially, you learn to take an image, and represent that image as a set of numbers (vector or matrix). For this example, we generate our image embedding using ResNet50 .


The model was originally trained to take an input image and predict the class of object in it—a decent surrogate for our task. We take the full pre-trained model and cut off the last few layers of the network, since we want its learned image embedding rather than its class prediction. Then we can pass all our images through the chopped network to obtain a fixed length vector representation of the image.

### Cosine Similarity

Once we get the embeddings of the images, the recommendation system can calculate the cosine similarities of the embeddings. This way, the recommender system's candidate generation is not limited by defined labels. As a result, we can get more diverse results from the recommender system. 

![TSNE](/img/tsne.png)

## Results

As you can see, there are a bit more variations in the recommended items. For the belt, the recommended items include a shoe that has similar braided style as the original belt. For the women’s under garment, the recommendation includes casual and ethnic dresses as well as other night gowns.

## Future Works

In the future, I’d like to explore developing a model using natural language processing of the product descriptions and combine the use of both text based model and image based model. Then, I’d like to try using GAN(Generative Adversarial Network) to take texts as input and images as output based on its learnings.  



