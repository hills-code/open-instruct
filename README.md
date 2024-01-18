# LLaMA-Pro Training 

This is a modified version of the allenai/open-instruct repository used for the LLaMA-Pro project. This branch diverged from main at commit 9ebcb58. This branch implements the following features that are not present in 9ebcb58:

* Tuning specific layers while freezing the other parameters.
* Use gradient checkpointing to train.

The remaining portion of this README contains instructions to replicate training of the LLaMA-Pro model.

You can find more details in the original [open-instruct](https://github.com/allenai/open-instruct).
## Replicating Training


### Set up environment

We provide a file containing a dump of our training environment.

You can install all required packages via
```bash
pip install -r requirements.txt
```

### Prepare data

You can use the following command to prepare the instruction dataset.
```bash
python open_instruct/reformat_datasets.py --dataset evol_codealpaca meta_math SlimOrca WizardLM_evol_instruct_V2_196k
```

### Launching Training

Then, edit the provided .sh files to set paths based on your own system's saved locations for checkpoints and data files. The example can be found in [finetune_codealpaca](scripts/finetune_codealpaca.sh).

**Tip**: If you want to train the specific layers, you can an argument `extend_layers` in the script to specify the layers for training.

You can use the following command to run instruction tuning (finetuning a pretrained model to follow instructions):

``` bash
bash scripts/finetune_codealpaca.sh
```
