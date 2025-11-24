from training.trainer import GoldTrainer


def main():
    # Khởi tạo Trainer
    # epochs=50: Học 50 vòng (nhưng có thể dừng sớm nếu EarlyStopping kích hoạt)
    # batch_size=32: Học 32 mẫu mỗi lần cập nhật trọng số
    trainer = GoldTrainer(epochs=50, batch_size=32)

    # Bắt đầu train
    trainer.train()


if __name__ == "__main__":
    main()