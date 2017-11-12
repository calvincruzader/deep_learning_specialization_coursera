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
      * Also, this means you need both $$a^{[l]}, a^{[l+2]} \in \mathcal{R^{m \times n}}$$ to add the residual block, so 'same' convolutions are used during these 'skip connection' steps 
        * if the dimensions don't match, then an option would be : $$g(W^{[l+2]}a^{[l+1]} + b^{[l+2]} + W_{s}a^{[l]})$$
  * Res blocks allows you to train **much** deeper neural nets
  * Training error keeps on going down as the number of layers increases, whereas in plain networks, the training error goes up after a certain point when the neural net has 'too many' layers 
    * takes care of the vanishing and exploding gradient problem 
  * Informal thoughts : skipping neurons skip any vanishing or exploding functionality that might happen in the very next layer, so that, just in case the neurons DO vanish/explode with that next function, there'll be a fallback of the next non-linearity that does not get impacted by vanishing/exploding gradients 

**Why do ResNets work?**
* ResNets provide a baseline such that the activation of the later layers cannot get worse, only as bad as that previous layer 
  * makes training error pretty much a non-increasing function of the number of layers in a NN 
    * doesn't hurt performance 
* paper: He et al., 2015. Deep Residual Networks for Image Recognition 

**1x1 Convolutions aka Networks in Networks**
* Lin et al., 2013, Network in Network
* focuses on the **number of channels** in a current layer so that you can mutate it (increase/decrease/mess with complexity) in the next layer 
* As mentioned, you can focus on getting fully connected networks on channels and get a nonlinearity to get a more complex function 
* In my own words:
    * doing 1x1 convolutions makes sense when you want to focus on the number of channels of a given layer. you can shrink/increase/maintain the size of the input channels by giving a smaller/larger/same size number of filters. 
    * doing 1x1 convolutions also makes sense when you want to feed the input into a nonlinear activation function by channel (giving a fc layer such that (number input channels convolved_by number of filters -> number output channels) $$n_{c_i} \circledast n_{f} = n_{c_i}$$ Given $$X$$ has dimensions $$(n_{h_i}, n_{w_i}, n_{c_i})$$ and there is a filter $$F$$ such that $$F$$ has dimensions $$(f , f, n_f)$$ where $$f \in \mathcal{R}^1, f < n_{h_i}, f < n_{w_i}$$ then $$X \circledast F = Z$$ where $$Z$$ has dimensions $$(n_{h_o}, n_{w_o}, n_{c_o})$$ and $$n_{c_o} = n_{f}$$.
    


**Inception Network Motivation**
* Szegedy et al., 2014. Going deeper with vonvolutions 
* underlying motivation: How does one choose which filter/kernel to use for a ConvNN? 
  * you can choose by doing an inception layer! perform a 1x1, 3x3, 5x5, maxpool all in one layer and stack the outputs up
  * let the NN learn whatever parameters it wants to use or whatever convolution filter sizes it wants 
    * the problem here is computational cost 
    * you can use 1x1 convolution to reduce the number of channels of a lyer to reduce the computational cost by (example used caused a reduction by an order of magnitude)
      * does shrinking down the representation size (many channels to much fewer channels) hurt performance? 
        * so long as you shrink down 'within reason' it doesn't hurt performance and saves a lot of computation 

**Inception Network** 
* Inception network is just an inception model repeated a bunch of times into a network 
![googlenet](http://img.blog.csdn.net/20160225155403967 =1000x400)

--------------
Practical Advice for Using ConvNets
--------------
**Using Open-Source Implementation**
* advantages: 
  * don't have to write up your own NNs
  * parameters in these open-source sources can be pretrained

**Transfer Learning**
* for your small training sets:
  * freeze parameters and train only your own custom softmax layer at the very end 
* for your larger training sets 
  * freeze less layers 
* for your very large data sets:
  * use the whole network as **initialization** , replaces random initialization this way 

**Data Augmentation**
* common to increase performance of a given system
  * in the field of computer vision, we can't get enough data!
* common data augmentation methods:
  * mirroring, random cropping, rotation, shearing, local warping 
  * color shifting: RGB (+20,20,+20) or (-20,+20,+20)
    * these RGB values are drawn from some probability distribution 
    * principles components analysis: PCA color augmentation
      * keep the overall color tint the same 
      * details in AlexNet 
    * implementing distortions 
      * have a CPU thread for doing distortions on mini batches 
      * have another CPU/GPU thread on training data 
        * can do both above in parallel 

**The State of Computer Vision** 
* data vs. hand-engineering 
  * we still need to have more data for image recognition 
* if you only have a little data, there's more hand-engineering going on 
* For deep learning, learning algorithms have two sources of knowledge:
  1. Labeled data
  2. Hand engineering (features/network architecture/other components/transfer learning) 
     * in the absense of lots of data, hand engineering works very well 

* Tips for doing well on benchmarks/winning competitions  
  * ensembling: train several NNs (3-15 networks) independently and average their outputs (will perform 1% or 2% better)
  * multi-crop at test time: run a classifier on multiple versions of test images and average the results 
    * 10 - crop: center, zoomed a little in each corner, then flip , then zoom again the 4 corners  







  



