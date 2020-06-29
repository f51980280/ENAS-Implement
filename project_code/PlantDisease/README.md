### Use below command for traing the enas 

```
python3 PlantDisease_enas.py --search-for  {macro, mirco}
```
#### Then you will see the training step, and every epoch will save the result model in ./model  

#### PlantDisease_enas.py
#### Set Config file in line.56~100
#### Build a PlantDisease_Dataset Pytorch Contructor
#### Use DataAugmentation in dataloader 
##### Image Augmentation
```
-> Gaussian Blur, rate=0.25, sigma=(0, 3)
-> Horizontal flip, rate=0.5
-> Rotate, angle=(-20, 20)
-> Image Dropout, rate=0.25
-> Add hue and saturation, value=(-10, 10)
```
