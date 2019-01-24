                                Learning Low-Dimensional Temporal Representations with Latent Alignments



1. Introduction.

This package includes the prototype MATLAB codes and data for experiments on the MSR Activity3D dataset, the ChaLearn dataset, and the UCF101 dataset, described in

	"Learning Low-Dimensional Temporal Representations with Latent Alignments"
	Bing Su and Ying Wu. 2019.

	"Learning Low-Dimensional Temporal Representations"
	Bing Su and Ying Wu. ICML, 2018.

----------------------------------------------------------------------

"Activity3D", "ChaLearn" tested under Windows Server 2012 R2 x64, Matlab R2015b.

"UCF101" tested under Linux Ubuntu 16.04.2, Matlab R2018a.





2. License & disclaimer.

    The codes and data can be used for research purposes only. This package is strictly for non-commercial academic use only.





3.  Usage & Dependency.

This package contains three subfolders containing codes and data for experiments on the MSR Activity3D dataset, the ChaLearn dataset, and the UCF101 dataset, respectively.
  
- Activity3D --- For the 120-dimensional frame-wide features "MSR-DailyActivity3D.mat" provided in "https://www.uow.edu.au/~leiw/" (L. Wang, J. Zhang, L. Zhou, C. Tang, and W. Li, ¡°Beyond covariance: Feature representation with nonlinear kernel matrices,¡± ICCV 2015), we use the proposed LT-LDA to reduce the dimension to 80 and employ Kernelized-COV (J. Cavazza, A. Zunino, M. San Biagio, and V. Murino, ¡°Kernelized covariance for action recognition,¡± ICPR 2016) to perform classification in the subspace.
  
  Dependency:
     (1) The code for the paper ¡°Beyond Covariance Representation¡±: https://www.uow.edu.au/~leiw/
     (2) SVM and Kernel Methods Matlab Toolbox: http://asi.insa-rouen.fr/enseignants/~arakoto/toolbox/index.html
     (3) Kernelized-COV: https://www.iit.it/research/39-research/lines/pattern-analysis-and-computer-vision/pavis-code/573-kernelized-covariance-for-action-recognition
  
  Usage:
     (a) Download dependency (1)(2)(3); Add the code "kercov.m" of dependency (3) to the folder (e.g. 'Kernel_representation_Code_for_release_v03') of dependency (1); 
     (b) Add the installation paths of the dependencies (1)(2) at the beginning of "run_me_KMAt3D.m":
         addpath(genpath('PATH\SVM-KM'));
         addpath(genpath('PATH\Kernel_representation_Code_for_release_v03'));
     (c) Run the main code in Matlab:
         main



- ChaLearn --- For the 100-dimensional frame-wide features provided in "https://bitbucket.org/bfernando/videodarwin" (described in "B. Fernando, E. Gavves, J. O. M., A. Ghodrati, and T. Tuytelaars,¡°Modeling video evolution for action recognition,¡± CVPR, 2015."), we use the proposed LT-LDA to reduce the dimension to 45 and employ rank pooling and the SVM classifier to perform classification in the subspace. Data available at './datamat/' and part of the code for rank pooling are adapted from the data and code by Basura Fernando "https://bitbucket.org/bfernando/videodarwin". We have provided an organized version of the features: https://pan.baidu.com/s/1mjkonfeJMojoUGnMNYpXpw

  Dependency:
     vlfeat-0.9.18
     liblinear-1.96
     libsvm-3.20
     Add or change the installation paths of the three packages at the beginning of "Evaluate_LTLDA.m" below "TODO add path"

  Usage:
     Download the folder "datamat" from "https://pan.baidu.com/s/1mjkonfeJMojoUGnMNYpXpw" and put it under this folder ("ChaLearn"); Run the main code in Matlab:
     Evaluate_LTLDA
     


- UCF101 --- For the 2048-dimensional deep frame-wide features extracted by the ResNeXt-101 model pre-trained on the Kinetics dataset (K. Hara, H. Kataoka, and Y. Satoh, ¡°Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet,¡± CVPR, 2018), we use the proposed LT-LDA to reduce the dimension to 500 and employ mean pooling or rank pooling and the SVM classifier to perform classification in the subspace.

  Dependency:
     liblinear-2.21
     libsvm-3.23
     ResNeXt-101 feature extraction and pre-trained model: https://github.com/kenshohara/video-classification-3d-cnn-pytorch

  Usage:
     (a) Use the code and pre-trained ResNeXt model in "https://github.com/kenshohara/video-classification-3d-cnn-pytorch" to extract features;
     (b) Reorganize the features to form the training and testing set according to the three splits of UCF101:
         UCF101ProcessFeatures_I3D_kinect.m: change the input feature path (output of step (a)) and the output feature path (reorganized feature format) at the beginning of this code below "%% TODO: set the feature path and the output path".
     (c) Learn the projection by LT_LDA:
         EvaluateLTLDA_kinect.m: set the split (1 to 3) at the beginning of this code below "% TODO: change the split_count here".
     (d) Perform classification with the original features and the learned low-dimensional representations:
         UCF_ltlda_acc_par.m: set the path of dependencies and set the split at the beginning of this code below "%% TODO: add path and change the split_count"





4. Notice

1) The default parameters in this package are adjusted on the datasets used in the paper. You may need to adjust the parameters when applying it on a new dataset. In some cases, you may need to normalize or centralize the input features in sequences. Please see the paper for details.

2) We utilized some toolboxes, data, and codes which are all publicly available. Please also check the license of them if you want to make use of this package.





5. Citations

Please cite the following papers if you use the codes:

1) Bing Su and Ying Wu, "Learning Low-Dimensional Temporal Representations with Latent Alignments," 2019.

2) Bing Su and Ying Wu, "Learning Low-Dimensional Temporal Representations," International Conference on Machine Learning, 2018.
