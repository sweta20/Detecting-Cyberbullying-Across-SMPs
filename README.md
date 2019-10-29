# Detecting-Cyberbullying-Across-SMPs

Abstract. Harassment by cyberbullies is a significant phenomenon on the social media. Existing works for cyberbullying detection have at least one of the following three bottlenecks. First, they target only one particular social media platform (SMP). Second, they address just one topic of cyberbullying. Third, they rely on carefully handcrafted features of the data. We show that deep learning based models can overcome all three bottlenecks. Knowledge learned by these models on one dataset can be transferred to other datasets. We performed extensive experiments using three real-world datasets: Formspring (˜12k posts), Twitter (˜16k posts), and Wikipedia(˜100k posts). Our experiments provide several useful insights about cyberbullying detection. To the best of our knowledge, this is the first work that systematically analyzes cyberbullying detection on various topics across multiple SMPs using deep learning based models and transfer learning.

## Dataset

The three datasets used in the paper can be downloaded from [here](https://drive.google.com/open?id=11RMLCSIAO3dWk9ejSkVYc5tQwwK5pquG).

Please download the dataset and unzip at data/.

We have also used two different kind of embeddings for initialization which can be found at the mentioned links.

- [Sentiment Specific word embeddings (SSWE)](http://ir.hit.edu.cn/~dytang/paper/sswe/embedding-results.zip)
- [GLoVe](https://nlp.stanford.edu/projects/glove/)


### Prerequisites

- Keras
- Tflearn
- Tensorflow
- Xgboost
- Sklearn
- Numpy

### Instructions to run

 - models.py : All the model architectures are defined in this file.
 - DNNs.ipynb : This notebook is responsible for training DNN models with three methods to initialize word embeddings.
 - TraditionalML.ipynb : The results from training ML models such as SYM, Naive Bayes, etc can be generated using this nnotebook.
 - Transfer Learning.ipynb : We used transfer learning to check if the knowledge gained by DNN models on one dataset can be    used to improve cyberbullying detection performance on other datasets. The code for the same is available in this file.

To know more about the architecture used and results, please read our paper [here](https://arxiv.org/pdf/1801.06482.pdf).


### Reproducibility Study 

1. Maral Dadvar, and Kai Eckert. "[Cyberbullying Detection in Social Networks Using Deep Learning Based Models; A Reproducibility Study.](https://arxiv.org/pdf/1812.08046.pdf)" arXiv preprint arXiv:1812.08046 (2018).
2. Aymé Arango, Jorge Pérez, and Barbara Poblete. "[Hate Speech Detection is Not as Easy as You May Think: A Closer Look at Model Validation.](https://users.dcc.uchile.cl/~jperez/papers/sigir2019.pdf)" Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 2019. 
3. Chris Emmery, Ben Verhoeven, Guy De Pauw, Gilles Jacobs, Cynthia Van Hee, Els Lefever, Bart Desmet, Véronique Hoste, Walter Daelemans. "[Current Limitations in Cyberbullying Detection: On Evaluation Criteria, Reproducibility, and Data Scarcity."](http://arxiv.org/abs/1910.11922) ArXiv:1910.11922 [Cs], Oct. 2019. arXiv.org. [[Code]](https://github.com/cmry/amica)

In [2] and [3], the authors discuss the limitations of our oversampling method (Section 4.2) in that, the way oversampling is currently handled may lead to overfitting. We found their claims/criticisms valid and important but we haven't conducted any new experiments to explicitely test and improve upon the limitations. Based on these studies, we no longer claim that our models provide state of the art results on the dataset untill further experiments are carried to study the effect of oversampling on the representations learned. We do believe that our paper and repository still serves as a useful resourse for modelling cyberbullying detection and understanding transferability between platforms. We suggest our viewers to look at these papers to be more aware of the limitations. 
 
