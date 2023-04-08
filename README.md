# BrainTalker: Low-Resource Brain-to-Speech Synthesis with Transfer Learning using Wav2Vec 2.0
Official Pytorch implementation of BrainTalker that submitted to the 2023 INTERSPEECH.
<br>

## Training :
```
python train.py --{Your experiment name} --{Number of gpu to allocate}
```
In `config.json`, change `trainset_folder` `path_testset_folder` to the absolute path of the dataset directory.<br>

## Note:
* Please download pre-trained wav2vec models from **[Facebook fairseq repository](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md)**. You can download 'wav2vec_small.pt' from Wav2Vec2.0 Base No finetunning tab.

