<h1>A Study on Prompt Sensitivity Using SAM</h1>
Authors:<br>
Alan(Xinglin) Zhong <br>
Aravind Thiagarajan <br>
Ayush Alpeshbhai Patel <br>

<h2> Introduction</h2>
This project investigates the robustness of training-free, prompt-guided object counting systems. By introducing controlled errors into bounding box, point, and text prompts, we evaluate how prompt noise affects counting accuracy and identify key failure modes in segmentation-based approaches.

<h2> Installation </h2>
1. The code requires python>=3.8, as well as pytorch>=1.7 and torchvision>=0.8. <br>
2. Please follow the instructions <a href="https://pytorch.org/get-started/locally/" target="_blank">here</a> to install both PyTorch and TorchVision dependencies. <br>
3. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

<h2> Getting Started </h2>
1. Download the <a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" target="_blank">'vit_b'</a> pre-trained model of SAM and save it to the folder 'pretrain'. <br>
2. Download the <a href="https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing" target="_blank">FSC-147</a> and <a href="https://datasets.activeloop.ai/docs/ml/datasets/carpk-dataset/" target="_blank">CARPK</a> datasets and save them to the folder 'dataset' <br>

<h2>Execution</h2>
Make sure you have all the requirements installed and dataset downloaded (At least the FSC147 dataset).
Then, navigate to dataset folder.
Run

```
python create_random_images.py
```
feel free to change the parameters in the script.

Then, back to the prject folder and run:
```
python main-fsc147.py --test-split='test' --prompt-type='box' --device='cuda:0'
```
or

```
python main-carpk.py --test-split='test' --prompt-type='box' --device='cuda:0'
```

<h2>Expected Output</h2>

```
20it [00:57,  2.88s/it]
MAE:5.50,RMSE:9.27,NAE:0.38,SRE:2.66
```
<h2> Acknowledgement </h2>
We are greatly thankful to our supervisor Dr.Garima Bajwa, for her unbounded support and guidance. <br>
We are greatful to Zenglin Shi, Ying Sun, Mengmi Zhang, original authors of research paper Training-free Object Counting with Prompts, for their hardworks<br>
You can find their research paper here <a href="https://openaccess.thecvf.com/content/WACV2024/papers/Shi_Training-Free_Object_Counting_With_Prompts_WACV_2024_paper.pdf" target="_blank">Training-free Object Counting with Prompts</a> <br>
And we are greatful to you, our readers, for your attentions and love!<br>