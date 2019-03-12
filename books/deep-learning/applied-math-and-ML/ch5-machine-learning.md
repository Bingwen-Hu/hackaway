# machine learning basic
machine learning is interesting because developing our understanding of machine learning entails developing our understanding of the principles that underlie intelligence.

## 5.1 machine learning tasks, T
+ classification
+ Classification with missing inputs: This kind of situation arises frequently in medical diagnosis, because many kinds of medical tests are expensive or invasive. One way to efficiently define such a large set of functions is to learn a probability distribution over all of the relevant variables, then solve the classification task by marginalizing out the missing variables.
+ Regression
+ Transcription: OCR, speech recognition
+ Machine translation
+ Structured output: One example is parsing—mapping a natural language sentence into a tree that describes its grammatical structure and tagging nodes of the trees as being verbs, nouns, or adverbs, and so on. Another example is pixel-wise segmentation of images, where the computer program assigns every pixel in an image to a specific category.
+ Anomaly detection
+ Synthesis and sampling: For example, in a speech synthesis task, we provide a written sentence and ask the program to emit an audio waveform containing a spoken version of that sentence.
+ Imputation of missing values
+ Denoising
+ Density estimation or probability mass function estimation

Of course, many other tasks and types of tasks are possible. The types of tasks we list here are intended only to provide examples of what machine learning can do, not to define a rigid taxonomy of tasks.

## 5.2 The Performance Measure, P
+ For tasks such as classification, classification with missing inputs, and transcription, we often measure the accuracy of the model.
+ We can also obtain equivalent information by measuring the error rate
+ For tasks such as density estimation, it does not make sense to measure accuracy, error rate, or any other kind of 0-1 loss.
+ In some cases, this is because it is difficult to decide what should be measured.
+ n other cases, we know what quantity we would ideally like to measure, but measuring it is impractical.

## 5.3 The Experience, E
Machine learning algorithms can be broadly categorized as unsupervised or supervised by what kind of experience they are allowed to have during the learning process.

Roughly speaking, unsupervised learning involves observing several examples of a random vector x, and attempting to implicitly or explicitly learn the probability distribution p(x), or some interesting properties of that distribution, while supervised learning involves observing several examples of a random vector x and an associated value or vector y, and learning to predict y from x, usually by estimating p(y|x).

Traditionally, people refer to regression, classification and structured output problems as supervised learning. Density estimation in support of other tasks is usually considered unsupervised learning.

One common way of describing a dataset is with a design matrix. A design matrix is a matrix containing a different example in each row.

In cases that any two example vectors have not the same size, rather than describing the dataset as a matrix with m rows, we will describe it as a set containing m elements: {x(1) , x(2) , . . . , x(m)}. 

## 5.4 Example: Linear Regression
We define input vector $x \in \R^n$ and output scalar $y \in R$. Let $\hat y$ be the value that our model predicts y should take on. 
$$ \large \hat y = w^T x$$
where w ∈ R n is a vector of parameters.

We thus have a definition of our task T: to predict y from x by outputting $ŷ = w^T x$. Next we need a definition of our performance measure P.
$$\Large MSE_{test} = \frac{1}{m} \sum \limits_{i} (\hat{y}^{(test)} - y^{(test)})_i^2$$
We can also see that
$$\Large MSE_{test} = \frac{1}{m} ||\hat{y}^{(test)} - y^{(test)}||_2^2$$

minimize the mean squared error on the training set, $MSE_train$.