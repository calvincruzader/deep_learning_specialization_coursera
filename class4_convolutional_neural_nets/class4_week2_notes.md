course4_week2_notes 
--------------
Case studies:
--------------

**Why look at case studies?**
* to gain intuition of how to build NNs
* plus, many convNet architectures can be transferred from one problem to another 

**Classic Networks** 
* LeNet-5: LeCun et al., 1998. Gradient-based learning applied to document recognition 
  * Goal: recoognize handwritten digits 
  * patterns that we still see today:
    * $$n_H$$, $$n_{W}$$ goes down while $$n_C$$ goes up as we go down the layers 
![LeNet](https://world4jason.gitbooks.io/research-log/content/deepLearning/CNN/img/lenet-result.png)
* AlexNet: Krizhevsky et al., 2012. ImageNet classification with deep convolutional neural networks 
  * similar to LeNet but much bigger, 60million parameters 
    * demonstrates scaling of a robust architecture despite much more data 
  * used ReLU
  * Local response normalization (LRN) (?)
![AlexNet_pic](https://world4jason.gitbooks.io/research-log/content/deepLearning/CNN/Model%20&%20ImgNet/alexnet/img/alexnet2.png)
* VGG-16: Simonyan & Zisserman 2015. Very deep convolutional networks for large-scale image recognition 
  * simplified neural network architectures 
  * all convolution layers had 3x3 filters, a stride of 1, 'same' convolution. All max_pool layers were 2x2 with a stride of 2         
    * number of filters for convolutions doubles from initial 64 until 512
  ![VGG_pic](https://www.cs.toronto.edu/~frossard/post/vgg16/vgg16.png)

**ResNets**
* vanishing and exploding gradients is a big problem, resnets take care of that 
* Residual Network
* Residual block 
  * $$a^{[l+2]} = g(z^{[l+2]} + a^{[l]}) = g(W^{[l+2]}a^{[l+1]} + b^{[l+2]} + a^{[l]})$$
  * if you're using $$L_2$$ regularization, $$W^{[l+2]}$$ will tend to shrink and if $$W^{[l+2]}=0 \Rightarrow g(a^{[l]}) = a^{[l]}$$
    * $$\therefore$$ the identity function is easy for the residual block to learn!
      * Also, this means you need both $$a^{[l]}, a^{[l+2]} \in \mathcal{R^{m,n}}$$ to add the residual block, so 'same' convolutions are used during these 'skip connection' steps 
        * if the dimensions don't match, then an option would be : $$g(W^{[l+2]}a^{[l+1]} + b^{[l+2]} + W_{s}a^{[l]})$$
  



