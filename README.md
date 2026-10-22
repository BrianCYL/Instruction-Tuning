## Preparing
Install the dependencies listed by running:
```
pip install -r requirement.txt
```

Download the dataset and checkpoints (if needed) from following command:
```bash
bash download.sh
```
Then, the folder structure should be similar to the one below:

```
|- data
    |- train.json
    |- test_public.json
    |- test_private.json
|- adapter_checkpoint
    |- adapter_config.json
    |- adapter_model.safetensors
```

## Training

### Gemma-2 Model
Pleace run the following command:
```
bash train.sh gemma
```
After running the above bash script, the trained model will be stored in a folder called `checkpoints_gemma`, which is shown below. 
Also, the json file of the evaluation perplexity is included in the checkpoint folder, which is shown below.
```
|- ...
|- data
|- train.sh
|- checkpoints_gemma
    |- adapter_config.json
    |- adapter_model.safetensors
    |- ...
    |- ppl_tracking.json
```

### Taiwan Llama 8B Model
Pleace run the following command:
```
bash train.sh llama
```
After running the above bash script, the trained model will be stored in a folder called `checkpoints_llama`, which is shown below. 
Also, the json file of the evaluation perplexity is included in the checkpoint folder, which is shown below.
```
|- ...
|- data
|- train.sh
|- checkpoints_llama
    |- adapter_config.json
    |- adapter_model.safetensors
    |- ...
    |- ppl_tracking.json
```

## Inference
Run the following command to generate translations with a trained LoRA adapter:
```
bash run.sh <base_model> <peft_model> <test_file> <output_file>
```
`<base_model>` is the original Hugging Face base model, such as `zake7749/gemma-2-2b-it-chinese-kyara-dpo` or `yentinglin/Taiwan-LLM-7B-v2.0-chat`.
`<peft_model>` is the directory containing the trained adapter checkpoint, such as `./checkpoints_gemma`.
`<test_file>` should be a JSON file containing `id` and `instruction` fields.
`<output_file>` is the output JSON file containing the generated `id` and `output` fields.
For example:
```
bash run.sh \
zake7749/gemma-2-2b-it-chinese-kyara-dpo \
./checkpoints_gemma \
./data/test_public.json \
./predictions_gemma.json
```