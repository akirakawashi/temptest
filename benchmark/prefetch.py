"""Загрузка весов GigaAM на CPU без доступа к записям и без inference."""
import os


def main():
    import gigaam

    root = os.environ.get("EMO_CACHE", "/cache/gigaam")
    for name in ("v3_e2e_rnnt", "emo"):
        print(f"Подготовка: загрузка весов {name}, записей и GPU нет", flush=True)
        model = gigaam.load_model(name, device="cpu", fp16_encoder=False, use_flash=False, download_root=root)
        del model


if __name__ == "__main__":
    main()
