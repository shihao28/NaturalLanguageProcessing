class Config(object):
    # ========================================================
    # Train Settings
    # ========================================================
    TRAIN = dict(
        model_checkpoint="t5-small",
        dataset_name="cnn_dailymail",
        dataset_config_name="3.0.0",
        text_column="article",
        summary_column="highlights",
        max_source_length=1024,
        max_target_length=128,
        seed=1010,
        batch_size=1
    )
    
    # ========================================================
    # Predict Settings
    # ========================================================
    PREDICT = dict(
        model_name="facebook/bart-large-cnn",
        tokenizer_name="facebook/bart-large-cnn",
        encode_max_length=512,
        encode_truncation=True,
        generate_min_length=50,
        generate_max_length=120
        
    )