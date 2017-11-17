course4_week3notes: Object Detection 

-------
Detection Algorithms
-------
**Object Localization**
* Recall that when building a ConvNet results in a set of features being fed into a softmax unit that outputs a predicted class ie a classification output. What if you wanted to localize the classification detected, then you would add localization numbers $$(b_x, b_y, b_h, b_w)$$ standing for (x_position, y_position, height, width) to parameterize the **bounding box** of the detected object. 

* Thus, the target label $$y$$ is defined/modifed as follows: $$y= \left( \begin{array}{} P_c \\ b_x \\ b_y \\ b_h \\ b_w \\ c_1 \\ c_2 \\ \vdots \end{array} \right)$$ where $$P_c$$ denotes the probability that any object is in the image, $$b_x, b_y, b_h, b_w$$ are the localization numbers that form the bounding box, and $$c_1, c_2,...$$ are the set of objects you want to classify.
  * This assumes that there is only one object to classify in the image 
  * Examples: Given that $$c_1$$ will denote the probability of a pedestrian is in the image, $$c_2$$ a car, $$c_3$$ a motorcycle, and $$c_4$$ a background (no object detected), then:
    * $$x_1=$$ ![car_image](http://www.publicdomainpictures.net/pictures/30000/velka/traveling-by-car.jpg =200x133)  $$\Rightarrow$$ $$y_1=\left( \begin{array}{} 1 \\ b_x \\ b_y \\ b_h \\ b_w \\ 0 \\ 1 \\ 0  \end{array} \right)$$,  $$x_2=$$   ![landscape_image](https://static.pexels.com/photos/371633/pexels-photo-371633.jpeg =200x133) $$\Rightarrow$$ $$y_2=\left( \begin{array}{} 0 \\ ? \\ ? \\ ? \\ ? \\ ? \\ ? \\ ?  \end{array} \right)$$
* The **loss function** can then be defined as: $$\mathcal{L}(\hat{y},y)=\begin{cases} (\hat{y}_1-y_1)^2 + (\hat{y}_2-y_2)^2 + ... + (\hat{y}_n-y_n)^2 \kern{1ex} \text{  if  }   y_1 = 1 \\  (\hat{y}_1-y_1)^2 \kern{32ex} \text{  if  } y_1 = 0 \end{cases}$$  
  * The loss function above is only an example. You can use different loss functions per section of the $$\hat{y}$$, that is: 
    * Log-likelihood for $$c_1,...,c_n$$ through softmax
    * Squared error for bounding boxes
    * Logistic regression loss for the $$P_c$$


**Landmark Detection**
* You can also just generalize object localization above to output $$(x,y)$$ coordinates of important points in an image, called landmarks, that you want the neural network to recognize. 
  * Let's call each landmark $$l_i$$ and, for an example let's have a face detector with 64 landmarks for different sections of a person's face.
    * ConvNet $$\Rightarrow \left( \begin{array}{} P_{\text{face}} \\ l_{1x} \\ l_{1y} \\ \vdots \\ l_{64x} \\ l_{64y}  \end{array} \right)$$ 
  * To train a network like this, you'll need a labeled training set of both x and y (uh oh) 
  * You could also have a pose detection, such as instead of just face features, entire body features, like the left elbow,  right knew, etc., and generate landmarks based off a perceived person in an image's pose.

**Object Detection** 
* closely cropped images in training set -> convnet -> y 
  * then use that convnet in sliding windows detection algorithm:
    * little window on a subsection of the image -> ConvNet -> {0,1}
    * shift window based in some stride and iterate throughout the image 
    * increase the size of the window then repeat steps above steps 
    * keep increasing the size of the window 
    * classify each region contained in each window as having a face or whatever or not   
    * ![object detection image](https://i.vimeocdn.com/filter/overlay?src0=https%3A%2F%2Fi.vimeocdn.com%2Fvideo%2F72275532_1280x1220.jpg&src1=https%3A%2F%2Ff.vimeocdn.com%2Fimages_v6%2Fshare%2Fplay_icon_overlay.png =200x200)
    * huge **disadvantage** is computational cost: you're using a ConvNet for each and every one of the iterations!
      * large strides would hurt performance, small strides would be computationally intensive 

**Convolutional Implementation of Sliding Windows**
* Basically, compute all window predictions at once! We do this by treating each window as it's own region to be convolved by   
* turn FC layers into convolutional layers (1x1xNum_Filter volumes) by doing convolutions on the layers that were supposed to be FC 
* Sermanet et al., 2014, OverFeat: Integrated recognition, localization and detection using convolutional networks 
* alot of the computation is duplicated, conv impl allows forward passes to share computation 
* why do we do two FCs?
* still, one disadvantage is that the bounding box is not going to be perfectly centered 

**Bounding Box Predictions** 
* Improve bounding box estimation 
* holy moly, **need to improve this note** 
* Redmon et al., 2015, You Only Look Once: Unified real-time object detection
* YOLO algorithm, works for real-time object detection!

**Intersection Over Union**
* How do you tell if your object detection algorithm is going well?
  * compute the Intersection over Union of ground truth bounding box and hypothesis bounding box 
    * a hypothesis bounding box is good if IoU $$\geq$$ 0.5 (or some other arbitrary hand-picked number)

**Non-Max Suppression**
* there is an issue with YOLO algorithm s.t. there may be multiple object detections in different grid cells for the same object 
* non-max suppression deals with this issue 