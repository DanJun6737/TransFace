# TransFace
This is the code of TransFace

## Requirements
* Install Pytorch (torch>=1.9.0)
* ```pip install -r requirement.txt```
 
## Datasets

You can download the training datasets, including MS1MV2 and Glint360K:

* MS1MV2: [Google Drive](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view)
* Glint360K: [Baidu](https://pan.baidu.com/share/init?surl=GsYqTTt7_Dn8BfxxsLFN0w) (code=:o3az)

You can download the test dataset IJB-C as follows:

* IJB-C: [Google Drive](https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view) 

## How to Train Models
To run on a machine with 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_baseline_vit_s.py 
```

## How to Test Models
To run on a machine with 8 GPUs:
```
python eval_ijbc_baseline.py --model-prefix work_dirs/glint360k_vit_s_baseline/model.pt --result-dir work_dirs/glint360k_vit_s_baseline --network vit_s_dp005_mask_0 > ijbc_glint360k_vit_s.log 2>&1 &
```


### TransFace Pretrained Models 

You can download the test models reported in our paper as follows:

1 Training Data: MS1MV2
* MS1MV2 TransFace-S: [Google Drive](https://drive.google.com/file/d/1UZWCg7jNESDv8EWs7mxQSswCMGbAZNF4/view?usp=share_link)
* MS1MV2 TransFace-B: [Google Drive](https://drive.google.com/file/d/16O-q30mH8d3lECqa5eJd8rABaUlNhQ0K/view?usp=share_link)
* MS1MV2 TransFace-L: [Google Drive](https://drive.google.com/file/d/1uXUFT6ujEPqvCTHzONsp6-DMIc24Cc85/view?usp=share_link)

2 Training Data: Glint360K
* Glint360K TransFace-S: [Google Drive](https://drive.google.com/file/d/18Zh_zMlYttKVIGArmDYNEchIvUSH5FQ1/view?usp=share_link)
* Glint360K TransFace-B: [Google Drive](https://drive.google.com/file/d/13IezvOo5GvtGVsRap2s5RVqtIl1y0ke5/view?usp=share_link)
* Glint360K TransFace-L: [Google Drive](https://drive.google.com/file/d/1jXL_tidh9KqAS6MgeinIk2UNWmEaxfb0/view?usp=share_link)


Accurcay of our models:

| Training Data | Model | GFLOPs | IJB-C(1e-6) | IJB-C(1e-5) | IJB-C(1e-4) | IJB-C(1e-3) | IJB-C(1e-2) | IJB-C(1e-1) | log |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| MS1MV2 | TransFace-S | 5.8 | 86.75 | 93.87 | 96.45 | 97.51 | 98.34 | 98.99 | link |
| MS1MV2 | TransFace-B | 11.5 | 86.73 | 94.15 | 96.55 | 97.73 | 98.47 | 99.11 | link |
| MS1MV2 | TransFace-L | 25.4 | 86.90 | 94.55 | 96.59 | 97.80 | 98.45 | 99.04 | link |

| Training Data | Model | GFLOPs | IJB-C(1e-6) | IJB-C(1e-5) | IJB-C(1e-4) | IJB-C(1e-3) | IJB-C(1e-2) | IJB-C(1e-1) | log |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Glint360K | TransFace-S | 5.8 | 89.93 | 96.06 | 97.33 | 98.00 | 98.49 | 99.11 | link |
| Glint360K | TransFace-B | 11.5 | 88.64 | 96.18 | 97.45 | 98.17 | 98.66 | 99.23 | link |
| Glint360K | TransFace-L | 25.4 | 89.71 | 96.29 | 97.61 | 98.26 | 98.64 | 99.19 | link |

### Installing
 
A step by step series of examples that tell you how to get a development env running
 
Say what the step will be
 
```
Give the example
```
 
And repeat
 
```
until finished
```
 
End with an example of getting some data out of the system or using it for a little demo
 
## Running the tests
 
Explain how to run the automated tests for this system
 
### Break down into end to end tests
 
Explain what these tests test and why
 
```
Give an example
```
 
### And coding style tests
 
Explain what these tests test and why
 
```
Give an example
```
 
## Deployment
 
Add additional notes about how to deploy this on a live system
 
## Built With
 
* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds
 
## Contributing
 
Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.
 
## Versioning
 
We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 
 
## Authors
 
* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)
 
See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
 
## License
 
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
 
## Acknowledgments
 
* Hat tip to anyone whose code was used
* Inspiration
* etc
