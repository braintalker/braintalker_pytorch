# BrainTalker: Low-Resource Brain-to-Speech Synthesis with Transfer Learning using Wav2Vec 2.0
Official Pytorch implementation of BrainTalker that submitted to the 2023 INTERSPEECH.
<br>

## Training :
```
python train.py --{Your experiment name} --{Number of gpu to allocate}
```
In `config.json`, change `trainset_folder` `path_testset_folder` to the absolute path of the dataset directory.<br>
