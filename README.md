# Transfer-Learning-with-CNN
Fine-tuning Convolutional Neural Networks.

In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest.

We’ll explore two different types of Transfer Learning in this assignment. The first approach is using a pre-trained CNN as a fixed feature extractor. In this technique, all layers of the CNN is frozen except for the last fully-connected layer. This last layer is changed to suit the task at hand. In this part of the assignment, you will take a pretrained model in PyTorch and replace the last fully connected layer which classifies images into 1000 classes into a new classifier that is adapted to classify images into 5 classes.

The second approach is to fine-tune the entire pre-trained network rather than just the final layer. Once again, you’ll replace the final fully connected layer of the network with a 5-class classifier. However, you’ll not freeze the rest of the weights of the CNN.

In both cases, you will take a small 2500 image dataset comprising 5 classes and train (rather fine-tune) the CNN on this small dataset. In your assignment submission, you’ll be giving us your trained models. You should train for at least 50 epochs.
