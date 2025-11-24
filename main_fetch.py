import os
from dotenv import load_dotenv
from data_loader.market_loader import MarketLoader
from data_loader.macro_loader import MacroLoader

# Load biến môi trường từ file .env
load_dotenv()


def main():
    print("=== BẮT ĐẦU QUÁ TRÌNH THU THẬP DỮ LIỆU ===")

    # 1. Lấy dữ liệu Thị trường (Yahoo)
    market_loader = MarketLoader()
    market_loader.fetch_data(start_date="2000-01-01")

    print("-" * 30)

    # 2. Lấy dữ liệu Vĩ mô (FRED)
    fred_key = os.getenv("FRED_API_KEY")

    if not fred_key:
        print("❌ LỖI: Không tìm thấy FRED_API_KEY trong file .env")
        return

    macro_loader = MacroLoader(api_key=fred_key)
    macro_loader.fetch_data(start_date="2000-01-01")

    print("=== HOÀN TẤT! KIỂM TRA THƯ MỤC data/raw ===")


if __name__ == "__main__":
    main()