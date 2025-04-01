# skyrisTest

Overall structure:
The approach for facial recognition was to utilize libraries to have access to the systems webcam, and to capture images of the detected face and run it through a classification neural network.
After having a general sense on what to build, I dove into researching what libraries to use and how to build a neural network. A simple sequential architecture is used for the network, taking in a 1x(48*48) size tensor as the input to the model. The model's output layer results in a tensor of size 1x7, with each entry corresponding to an emotion. The emotion is chosen on which entry has the highest activation value, and is then displayed right next to the detected face image on the camera.

Through research, the project is developed using OpenCV, DeepFace and PyTorch. The FER2013 dataset obtained from Kaggle and is used to train the model. DeepFace was used just to get a feel on utilizing API calls on pretrained models and to jump start on the initialization of the project, thinking about on how should the neural network be structured. 

Throughout the building process, there were times where I felt lost on how to proceed, especially when building the network and training the model. Most of the research time was spent on reading the PyTorch documentation to figure out the tools needed. 

DeepFace's model preformed much more accurately than my own network. Perhaps further expansions could be to dig into a wider variety of emotions and not just the basic ones listed by the dataset. 