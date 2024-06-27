<p align="center">
  <br><br>
  <img src="model_arch.png" width="1300" height="600"><br>
</p>


-----------------------------------------------------------
<!-- 1. How to run our main notebook
2. Explain folder structure
3. Explain what code we utilized  -->

## General Information
Our project utilizes the "Final_Experiment_Notebook.ipynb" jupiter notebook as a format for all experiments. 
The experiement notebooks have been setup to run in a colab environment. 
You must also install the flickr8k dataset in a file folder called "archive" as seen in the folder structure below.
  
## Acknowledgement of Code Use
Our notebook has been adapted from the tensorflow image captioning [tutorial](https://www.tensorflow.org/tutorials/text/image_captioning). We originally tried to write our own code version of this in the model folder. As seen we ended up using the notebook format as this code was much more complex and enmeshed then originally envisioned. 

We also referenced this [Towards Data Science](https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa) article.

# Running the Experiments
To run the experiment, you can simply run the notebook top to bottom with the setup mentioned above. Note that if you run a particular experiment for a given feature extraction you must uncomment this code block

```
#not run once done, run only if different model for feature extraction
extract_features(train_dataset, 'train_cacheResnet101V2', mobilenet, tokenizer)
extract_features(test_dataset, 'test_cacheResnet101V2', mobilenet, tokenizer)
```

The above will run the feature extractor for the "Resnet101V2" model. You have to do this ONCE per experiment folder in "Stage2". This is because each folder in experiment 2 uses the same image feature extractor.


## Folder Structure

FINALPROJECT_DL  
 ┣ archive  
 ┃ ┣ Flickr8K_Dataset  
 ┃ ┣ Flickr8K_text  
 ┣ model  
 ┃ ┣ caption_decoder.py  
 ┃ ┣ caption_encoder.py  
 ┃ ┣ dataprep8k.py  
 ┃ ┣ object_detector.py  
 ┃ ┗ PYTORCH_image_encoder.py  
 ┣ old  
 ┃ ┣ datahandler.py  
 ┃ ┣ data_preprocessor.py  
 ┃ ┣ hchandak3.ipynb  
 ┃ ┣ Main.ipynb  
 ┃ ┗ vkoti7.ipynb  
 ┣ Stage2  
 ┃ ┣ mobilenet  
 ┃ ┃ ┣ checkpoint  
 ┃ ┃ ┣ mobile-Causal-cross.ipynb  
 ┃ ┃ ┣ mobile-Causal-multicross.ipynb  
 ┃ ┃ ┣ mobile-Causal.ipynb  
 ┃ ┃ ┣ mobile-cross.ipynb  
 ┃ ┃ ┣ mobile-multicross.ipynb  
 ┃ ┃ ┣ model.init.data-00000-of-00001  
 ┃ ┃ ┗ model.init.index  
 ┃ ┣ Resnet101  
 ┃ ┃ ┣ Resnet101-Causal-Cross.ipynb  
 ┃ ┃ ┣ Resnet101-Causal-MultiCross.ipynb  
 ┃ ┃ ┣ Resnet101-Causal.ipynb  
 ┃ ┃ ┣ Resnet101-Cross.ipynb  
 ┃ ┃ ┗ Resnet101-MultiCross.ipynb  
 ┃ ┣ ResNet50  
 ┃ ┃ ┣ ResNet50-Causal-cross.ipynb  
 ┃ ┃ ┣ ResNet50-Causal-multicross.ipynb  
 ┃ ┃ ┣ ResNet50-Causal.ipynb  
 ┃ ┃ ┣ ResNet50-Cross.ipynb  
 ┃ ┃ ┗ ResNet50-Multicross.ipynb  
 ┃ ┣ VGG16  
 ┃ ┃ ┣ VGG16-Causal-cross.ipynb  
 ┃ ┃ ┣ VGG16-Causal-multicross.ipynb  
 ┃ ┃ ┣ VGG16-Causal.ipynb  
 ┃ ┃ ┣ VGG16-Cross.ipynb  
 ┃ ┃ ┗ VGG16-Multicross.ipynb  
 ┃ ┗ VGG19  
 ┃ ┃ ┣ VGG19-Causal-cross.ipynb  
 ┃ ┃ ┣ VGG19-Causal-multicross.ipynb  
 ┃ ┃ ┣ VGG19-Causal.ipynb  
 ┃ ┃ ┣ VGG19-Cross.ipynb    
 ┃ ┃ ┗ VGG19-Multicross.ipynb  
 ┣ VGGStage-1    
 ┃ ┣ Stage1_round2_VGG16_3head_256.ipynb  
 ┃ ┣ Stage1_round2_VGG16_3head_512.ipynb  
 ┃ ┣ VGG16-droput0.1-layer_head2.ipynb    
 ┃ ┣ VGG16-droput0.2-layer_head2.ipynb  
 ┃ ┣ VGG16-droput0.3-layer_head2.ipynb  
 ┃ ┣ VGG16-droput0.5-layer_head1.ipynb  
 ┃ ┣ VGG16-droput0.5-layer_head3-unit512.ipynb  
 ┃ ┣ VGG16-droput0.5-layer_head3.ipynb  
 ┃ ┣ VGG16-lr1e3.ipynb  
 ┃ ┣ VGG16-lr1e5.ipynb  
 ┃ ┣ VGG16-unit128.ipynb  
 ┃ ┣ VGG16-unit512-dropout0.3.ipynb  
 ┃ ┣ VGG16-unit512.ipynb  
 ┃ ┗ VGG16.ipynb  
 ┣ .gitignore  
 ┣ checkpoint  
 ┣ Final_Experiment_Notebook.ipynb  
 ┣ main.py  
 ┣ model.init.data-00000-of-00001  
 ┣ model.init.index  
 ┣ model_arch.png  
 ┣ README.md  
 ┣ requirements.txt    
 ┗ training_utils.py    





<!-- ## Code References  -->