import argparse
import sys
import os
import logging
import time

# 1. THIẾT LẬP ĐƯỜNG DẪN
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 2. IMPORT MODULES
from src.utils.config_utils import load_settings, setup_logging
from src.data_loader import MarketLoader, MacroLoader
from src.processing import DataProcessor
from src.training import ModelTrainer
from src.prediction import GoldPredictor
from src.backtesting import Backtester
from src.visualization import Visualizer
from src.evaluation import ModelEvaluator

# Khởi tạo logger
logger = logging.getLogger("MainController")


def run_fetch(settings):
    """Bước 1: Tải dữ liệu"""
    logger.info("📡 [1/6] BẮT ĐẦU TẢI DỮ LIỆU...")
    try:
        market_loader = MarketLoader(settings)
        start_date = settings['data'].get('start_date', '2000-01-01')
        market_loader.fetch_data(start_date=start_date)

        macro_loader = MacroLoader(settings)
        macro_loader.fetch_data(start_date=start_date)
        logger.info("✅ Tải dữ liệu hoàn tất.")
    except Exception as e:
        logger.error(f"❌ Lỗi tải dữ liệu: {e}")
        raise


def run_process(settings):
    """Bước 2: Xử lý dữ liệu"""
    logger.info("⚙️ [2/6] BẮT ĐẦU XỬ LÝ DỮ LIỆU...")
    try:
        processor = DataProcessor(settings)
        save_path = processor.run()
        logger.info(f"✅ Xử lý hoàn tất. File: {save_path}")
    except Exception as e:
        logger.error(f"❌ Lỗi xử lý: {e}")
        raise


def run_train(settings):
    """Bước 3: Huấn luyện Model"""
    logger.info("🏋️ [3/6] BẮT ĐẦU HUẤN LUYỆN...")
    try:
        trainer = ModelTrainer(settings)
        model_path = trainer.train()
        logger.info(f"✅ Huấn luyện hoàn tất. Model: {model_path}")
    except Exception as e:
        logger.error(f"❌ Lỗi huấn luyện: {e}")
        raise


def run_predict(settings):
    """Bước 4: Dự đoán tương lai"""
    logger.info("🔮 [4/6] BẮT ĐẦU DỰ ĐOÁN...")
    try:
        predictor = GoldPredictor(settings)
        result = predictor.predict()
        logger.info("✅ Dự đoán hoàn tất.")
    except Exception as e:
        logger.error(f"❌ Lỗi dự đoán: {e}")
        raise


def run_backtest(settings):
    """Bước 5: Kiểm thử chiến lược"""
    logger.info("💸 [5/6] BẮT ĐẦU BACKTEST...")
    try:
        bot = Backtester(settings)
        bot.run()
        logger.info("✅ Backtest hoàn tất.")
    except Exception as e:
        logger.error(f"❌ Lỗi Backtest: {e}")
        raise


def run_visualize(settings):
    """Bước 6: Vẽ biểu đồ"""
    logger.info("🎨 [6/6] BẮT ĐẦU VẼ BIỂU ĐỒ...")
    try:
        viz = Visualizer(settings)
        # viz.plot_forecast()
        # viz.plot_test_results()
        viz.plot_test_simulation()
        logger.info("✅ Vẽ biểu đồ hoàn tất.")
    except Exception as e:
        logger.error(f"❌ Lỗi Visualize: {e}")
        raise


def run_pipeline(settings):
    """
    🚀 CHẠY TOÀN BỘ QUY TRÌNH TỰ ĐỘNG (PIPELINE)
    """
    logger.info("\n" + "=" * 50)
    logger.info("🚀 BẮT ĐẦU CHẠY TOÀN BỘ HỆ THỐNG (FULL PIPELINE)")
    logger.info("=" * 50 + "\n")

    start_time = time.time()

    try:
        # Chạy lần lượt từng bước. Nếu bước trước lỗi, sẽ dừng ngay lập tức.
        run_fetch(settings)
        print("-" * 30)

        run_process(settings)
        print("-" * 30)

        run_train(settings)
        print("-" * 30)

        run_evaluate(settings)
        print("-" * 30)

        run_predict(settings)
        print("-" * 30)

        run_backtest(settings)
        print("-" * 30)

        run_visualize(settings)

        duration = time.time() - start_time
        logger.info("\n" + "=" * 50)
        logger.info(f"🏆 HOÀN THÀNH TẤT CẢ TÁC VỤ! Tổng thời gian: {duration:.2f} giây")
        logger.info("=" * 50)

    except Exception as e:
        logger.critical(f"🔥 QUY TRÌNH BỊ NGẮT DO LỖI: {e}")
        sys.exit(1)

def run_evaluate(settings):
    """Bước phụ: Đánh giá hiệu suất chi tiết"""
    logger.info("📊 [Evaluate] ĐÁNH GIÁ MÔ HÌNH...")
    try:
        evaluator = ModelEvaluator(settings)
        evaluator.run()
        logger.info("✅ Đánh giá hoàn tất.")
    except Exception as e:
        logger.error(f"❌ Lỗi đánh giá: {e}")


def main():
    parser = argparse.ArgumentParser(description="Gold Price Forecast Professional System")

    # Thêm lựa chọn 'pipeline' vào danh sách
    parser.add_argument('mode', type=str,
                        choices=['fetch', 'process', 'train', 'predict', 'backtest', 'visualize', 'pipeline', 'evaluate'],
                        help="Chọn chế độ chạy. Chọn 'pipeline' để chạy tất cả.")

    parser.add_argument('--config', type=str, default='config/settings.yaml', help="Đường dẫn config")

    args = parser.parse_args()

    setup_logging()

    try:
        settings = load_settings(args.config)
        logger.info(f"🤖 Hệ thống khởi động. Mode: {args.mode.upper()}")

        if args.mode == 'fetch':
            run_fetch(settings)
        elif args.mode == 'process':
            run_process(settings)
        elif args.mode == 'train':
            run_train(settings)
        elif args.mode == 'predict':
            run_predict(settings)
        elif args.mode == 'backtest':
            run_backtest(settings)
        elif args.mode == 'visualize':
            run_visualize(settings)
        elif args.mode == 'pipeline':
            run_pipeline(settings)
        elif args.mode == 'evaluate':
            run_evaluate(settings)

    except Exception as e:
        logger.critical(f"🔥 LỖI NGHIÊM TRỌNG HỆ THỐNG: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()