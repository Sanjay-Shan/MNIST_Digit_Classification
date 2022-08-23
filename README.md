## MNIST Digit Classification

The Project was done as a part of the class Assignment.The task was to perform the MNIST Digit classification using the given dataset.
All the outputs that were obtained have been placed in the output folder for reference ( except for the trained models).  

The following was expected out of students:

### Task1:
1) Initially flattened the input ie 28x28x1 to 784x1 ,so as to pass it through the FC layers
2) followed by a 100 neuron layer and finally a softmax classifier for multi class classification
3) To be noted that a Tensorboard summary writer was utilised to plot the accuracy and loss curve
4) All the FC layers in this model were accompanied by Sigmoid activation function.

Observation:
  There was a continous improvement in the accuarcy of the model until 44 the epoch after which the accuracy was constant.
  The training went on smoothly and quickly as it was a very small network.
  
### Task2:
1) Along with all the layers present in the last task ,2 additional conv layers were added with the mentioned configurations.
2) The computations of the input and the output feature shapes were made using (N-F/S)+1
3) Finally a minor changes was made in the input shape of the first FC as per the output of the flattened conv layer.

Observation:
  There was a significant improvement until 17th epoch after the accuarcy was constant.
  The number was epochs was set to 60.
  This one took a slightly higher time as compared to the first one.
  
### Task3:
1) The only changes that was made in this step was the change in the activation function used for the FC layers.
This network was run for only 15 iteration as the accuracy was really good.

Observation:
  The accuracy was around 99% on the test set.
  
### Task4:
1) An additional FC layer with a 100 neurons were added to the existing network.

Observation:
  There were a lot of spikes in the accuracy curve initially which later got stagnated.
  The last improvement in the accuracy was seen at 32nd epoch
  
### Task 5:
1) All the FC layers which had 100 neurons initially now were changed to 1000 Neurons.
2. Use Dropouts if necessary.

  Observation:
  There were a lot of fluctuations in the accuracy curve until the last epoch.
  But it gave a good accuracy.
  
Other observations:
1) Output changes for every single trial due to the random initialization of weights.( hence a phenomenon like seeding can get the same output every trial if it is for the
same dataset.
2) various other initialisations can be used like xaviour initilisation etc where the range of weights depends upon the number of nodes in the layer. (-1/sqrt(n) ,1/sqrt(n

Structure of the output folders for each of the tasks ,mentioned above:
1. Accuracy plot
2. Loss Plot
3. Trace file for training
