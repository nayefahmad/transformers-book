# # Domain-adaptation of a pretrained roberta model

# This example won't run without the dataset, but it shows how the process would work.

run_example = False
if run_example:

    # Main script:
    from transformers import (
        RobertaForMaskedLM,
        RobertaTokenizer,
        LineByLineTextDataset,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
    )


    path = '/huggingface/models/RobertaLM'
    model = RobertaForMaskedLM.from_pretrained(path)

    path = '/huggingface/tokenizers/roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(path)


    file_path = "language_model_train_data_combined_for_train.csv"
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=512, # largest token length supported by LM to be trained
    )

    file_path = "language_model_train_data_combined_for_test.csv"
    eval_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=512,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )


    save_dir = "."
    training_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        num_train_epochs=20,
        per_device_train_batch_size=24,
        logging_steps=2000,
        save_steps=2000,
        save_total_limit=3,
        seed=1
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model(save_dir)