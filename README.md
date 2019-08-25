## Intro
The repository for NAACL 2019 paper "Relation Extraction with Temporal Reasoning Based on Memory Augmented Distant Supervision".

We use torch == 0.4 for all the experiments. Also, 

1. TempMEM used to be called MemCNN. Most of the experiment is done by mem_cnn and mem_pcnn. 

Usage: examples for running experiments can be found in scripts. Due to my experience, better results come from the settings in "wiki_temp_mem_pe_epoch_70_lr_5e-3.sh" script. You can explore other choices. If you want to use miml and other bash scripts that use the NYT-10 dataset, please unzip the nyt.zip file in origin_data. Enjoy :)

Also, I'm still constantly updating the code, if you find any problem please raise an issue. Thanks! 

## Citation

If you use the code base, please cite the following paper.

```
 @inproceedings{
  yan2019relation,
  title={Relation Extraction with Temporal Reasoning Based on Memory Augmented Distant Supervision},
  author={Yan, Jianhao and He, Lin and Huang, Ruqin and Li, Jian and Liu, Ying},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={1019--1030},
  year={2019}
}
```
 

