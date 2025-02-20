import pytest
import os
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_pipeline import ModelPipeline
from src.utils import load_obj
from torch.utils.data import Dataset
import torch


@pytest.fixture
def data_ingestion():
    """Veri alma işlemini çalıştırır ve test için gerekli konfigürasyonu döndürür."""
    data_ingestion_pipeline = DataIngestionPipeline(TEST_MODE=True)
    config = data_ingestion_pipeline.dataingestionconfig
    data_ingestion_pipeline.run_data_ingestion_pipeline()
    return config


def test_data_ingestion(data_ingestion):
    """Veri ayrıştırma işleminin doğru yapıldığını test eder."""
    config = data_ingestion

    # Klasörlerin varlığını test et
    assert os.path.exists(config.train_data_path)
    assert os.path.exists(config.test_data_path)
    assert os.path.exists(config.valid_data_path)

    # Label klasörlerinin sayılarının tutması lazım
    assert len(os.listdir(config.train_data_path)) == sum(len(os.listdir(os.path.join(config.all_data_save_path, images_path))) for images_path in os.listdir(config.all_data_save_path))
    assert len(os.listdir(config.valid_data_path)) == sum(len(os.listdir(os.path.join(config.all_data_save_path, images_path))) for images_path in os.listdir(config.all_data_save_path))
    assert len(os.listdir(config.test_data_path)) == sum(len(os.listdir(os.path.join(config.all_data_save_path, images_path))) for images_path in os.listdir(config.all_data_save_path))

    # İçindeki görüntü sayıları test ediliyor
    img_num_train = sum(len(os.listdir(os.path.join(config.train_data_path, label))) for label in os.listdir(config.train_data_path))
    img_num_valid = sum(len(os.listdir(os.path.join(config.valid_data_path, label))) for label in os.listdir(config.valid_data_path))
    img_num_test = sum(len(os.listdir(os.path.join(config.test_data_path, label))) for label in os.listdir(config.test_data_path))
    
    all_data = img_num_valid+img_num_train+img_num_test
    
    assert img_num_test == int(all_data * config.test_split_rate)
    assert img_num_valid == int(all_data * config.valid_split_rate)

@pytest.fixture
def data_transformation():
    """Veri dönüşüm işlemini çalıştırır ve test için gerekli datasetleri döndürür."""
    data_transformation_pipeline = DataTransformationPipeline()
    config = data_transformation_pipeline.datatransformationconfig
    data_transformation_pipeline.run_data_transformation()

    train_dataset = load_obj(config.transformed_train_dataset)
    valid_dataset = load_obj(config.transformed_valid_dataset)
    test_dataset = load_obj(config.transformed_test_dataset)

    return config, train_dataset, valid_dataset, test_dataset


def test_data_transformation(data_transformation):
    """Veri dönüşüm aşamasının doğru çalıştığını test eder."""
    config, train_dataset, valid_dataset, test_dataset = data_transformation

    # Dataset tip kontrolü
    assert isinstance(test_dataset, Dataset)
    assert isinstance(test_dataset[0][0], torch.Tensor)

    # Verinin (image, label) şeklinde olmasını test et
    for dataset in [train_dataset, valid_dataset, test_dataset]:
        assert len(dataset[0]) == 2

        # Görüntü boyutu kontrolü
        C, H, W = dataset[0][0].shape
        assert C == config.channel_size
        assert H == config.img_resize_size and W == config.img_resize_size


@pytest.fixture
def model():
    """Model oluşturma pipeline'ını çalıştırır ve model objesini döndürür."""
    model_pipeline = ModelPipeline()
    config = model_pipeline.modelconfig
    model_pipeline.run_model_creating()

    return config, load_obj(config.model_save_path)


def test_model(model, data_transformation):
    """Modelin doğru çalışıp çalışmadığını test eder."""
    config, model = model
    _, _, _, test_dataset = data_transformation

    # Test verisini modele veriyoruz
    sample_input = test_dataset[0][0].unsqueeze(0)
    out = model(sample_input)

    # Modelin çıktısının shape'ini test et
    assert out.shape[1] == config.label_size
    assert out.shape[0] == 1  # batch_size = 1