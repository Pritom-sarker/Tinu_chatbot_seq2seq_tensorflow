# **Tinu_chatbot_seq2seq_tensorflow**

This is a  implementation of Twitter/Cornell-Movie Chatbot **Tinu** .
A sequence2sequence chatbot implementation with TensorFlow 1.10 .
I just use 10% data of full dataset. you can use full dataset.Then you need to read full article.

# Prerequisites

* tensorflow - 1.10
* numpy
* pyttsx3
* pypiwin32


# How to use:

1. Install Prerequisites
    > pip install -r requirements.txt
    
2. Create dataset
    > python final_data.py

3. Train Model
    > python train.py

4.  Test Model
    > python test_model.py

# Extra Features
**You can add more frequently asked questions and answers on a text file under the store folder.**


### Using a big Dataset you need to create your dataset using this steps and then follow the earlier instruction for training and testing

1. Download Twitter/Cornell-Movie Dataset [link](http://www.cs.cornell.edu/~cristian/memorability_files/cornell_movie_quotes_corpus.zip)

2. Create a folder named raw_data and  paste unzipped data there.

3. Run
    > python data.py

4. Run
    > python prep_data.py
5. Run
    > python final_data.py
    
    
  
# Extra instructions

 **Cost or loss will be saved in a loss file named  log.txt**

# Test Result of Tinu 

  you :: what is your name
  Tinu :: my name is tinu
  You :: who are you  
  Tinu :: a sequence2sequence chatbot
  You :: how  are you
  Tinu :: do you know who that is .
  
  
  
 **To make the chatbot usefull use a big dataset and train the model for a long time**
