import yaml
import os
import logging
import logging.config


def load_settings(config_path: str = "config/settings.yaml") -> dict:
    """
    Tải cấu hình từ file YAML.
    :param config_path: Đường dẫn tới file settings.yaml
    :return: Dictionary chứa cấu hình
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    return settings


def setup_logging(config_path: str = "config/logging.yaml", default_level=logging.INFO):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"📁 Đã tạo thư mục log: {log_dir}")

    # 2. Tải cấu hình logging
    if os.path.exists(config_path):
        with open(config_path, 'rt', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                logging.info("Logging đã được thiết lập từ file yaml.")
            except Exception as e:
                print(f"Lỗi khi tải file config logging: {e}")
                logging.basicConfig(level=default_level)
    else:
        logging.basicConfig(level=default_level)
        print("Không tìm thấy logging.yaml, sử dụng cấu hình mặc định.")