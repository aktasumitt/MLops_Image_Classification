class Configs():
    
    # After data ingestion
    DATA_LOCATION_PATH = "C:/Users/umtak/Downloads/archive.zip"
    ALL_DATA_SAVE_PATH = "all_data"
    TRAIN_DATA_PATH = "local_data/train/"
    VALID_DATA_PATH = "local_data/valid/"
    TEST_DATA_PATH = "local_data/test/"
    EXAMPLE_DATA_FOR_PYTEST = "test/images.zip" # bu path pytest için testte kullanılan datanın pathi


    # After transformation
    TRANSFORMED_TRAIN_DATASET_PATH = "artifacts/data_transformed/train_dataset.pkl"
    TRANSFORMED_TEST_DATASET_PATH = "artifacts/data_transformed/test_dataset.pkl"
    TRANSFORMED_VALID_DATASET_PATH = "artifacts/data_transformed/valid_dataset.pkl"

    # After creating model
    MODEL_SAVE_PATH = "artifacts/model/model.pkl"

    # After training
    CHECKPOINT_SAVE_PATH = "callbacks/checkpoints/checkpoint_last.pth.tar"
    SAVE_TRAINING_RESULT_PATH = "results/train_results.json"
    FINAL_MODEL_SAVE_PATH = "callbacks/final_model/final_model.pkl"
    
    # After Testing
    TESTED_MODEL_SAVE_PATH = "callbacks/tested_model/tested_best_model.pkl"
    SAVE_TESTING_RESULT_PATH = "results/test_results.json"
    BEST_CHECKPOINT_PATH = "callbacks/checkpoints/checkpoint_5-epoch.pth.tar"  # change this as your results
    
    # After prediction
    SAVE_PREDICTION_RESULT_PATH = "predict_artifact/results/result.json"
    PREDICTION_DATA_PATH= "predict_artifact/images"
    
    