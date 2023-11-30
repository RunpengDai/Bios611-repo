# record the BUGs I met in the process of learning

## 1. NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.
    solution: update the datasets package to the latest version

## 2. RuntimeError: expected scalar type Half but found Float
    solution: with torch.autocast("cuda"): 
                    trainer.train()
    
## 3. safetensors_rust.SafetensorError: Error while deserializing header: InvalidHeaderDeserialization
    - some bug about safetensors, default not save as safetensors. I just force it to save a bin file.
    solution: set save_safetensors=False in transformers.TrainingArguments

## 4. try to understand the loss calculataion in finetuning LLM, resulting in digging into the transformers trainer module
    - There is a line in __inner_training_loop_ :    
    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
     which prevents me from printing stuff during the training process
    - So I use a txt file to get the loss, label
    - It turns out that the loss is calculated within the Lamma2 model itself, which make sense because is a auto regressive. So still I don't know how the loss is computed.
