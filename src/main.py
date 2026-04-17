import os

from io_helpers import get_project_root, ensure_dir
from componentExtraction import process_7circles, process_covid_chart
from deblurring import process_building, process_dog
from denoising import process_text, process_rocket, process_wind_chart
from visualEnhancement import process_newspaper, process_name_plate


def build_required_processed_dirs(root: str) -> None:
    dirs = [
        os.path.join(root, "data", "processed", "blurEnhancement", "building"),
        os.path.join(root, "data", "processed", "blurEnhancement", "dog"),
        os.path.join(root, "data", "processed", "componentExtraction", "7circles"),
        os.path.join(root, "data", "processed", "componentExtraction", "COVID-19Chart"),
        os.path.join(root, "data", "processed", "noiseRemoval", "rocket"),
        os.path.join(root, "data", "processed", "noiseRemoval", "text"),
        os.path.join(root, "data", "processed", "noiseRemoval", "windChart"),
        os.path.join(root, "data", "processed", "visualEnhancement", "namePlate"),
        os.path.join(root, "data", "processed", "visualEnhancement", "newsPaper"),
        os.path.join(root, "reports", "figures"),
    ]

    for d in dirs:
        ensure_dir(d)


def main():
    root = get_project_root()
    build_required_processed_dirs(root)

    original_dir = os.path.join(root, "data", "original")

    paths = {
        "7circles": os.path.join(original_dir, "componentExtraction", "7circles.png"),
        "COVID-19Chart": os.path.join(original_dir, "componentExtraction", "COVID-19Chart.jpeg"),
        "building": os.path.join(original_dir, "blurEnhancement", "building.jpg"),
        "dog": os.path.join(original_dir, "blurEnhancement", "dog.jpeg"),
        "text": os.path.join(original_dir, "noiseRemoval", "text.jpeg"),
        "rocket": os.path.join(original_dir, "noiseRemoval", "rocket.jpeg"),
        "windChart": os.path.join(original_dir, "noiseRemoval", "windChart.png"),
        "newsPaper": os.path.join(original_dir, "visualEnhancement", "newsPaper.jpg"),
        "namePlate": os.path.join(original_dir, "visualEnhancement", "namePlate.jpg"),
    }

    print("=" * 60)
    print("Starting Mini Project Image Processing")
    print("=" * 60)

    # print("\n[1/9] Processing 7circles...")
    # process_7circles(paths["7circles"])

    # print("[2/9] Processing COVID-19Chart...")
    # process_covid_chart(paths["COVID-19Chart"])

    # print("[3/9] Processing building...")
    # process_building(paths["building"])

    print("[4/9] Processing dog...")
    process_dog(paths["dog"])

    # print("[5/9] Processing text...")
    # process_text(paths["text"])

    # print("[6/9] Processing rocket...")
    # process_rocket(paths["rocket"])

    # print("[7/9] Processing windChart...")
    # process_wind_chart(paths["windChart"])

    # print("[8/9] Processing newsPaper...")
    # process_newspaper(paths["newsPaper"])

    # print("[9/9] Processing namePlate...")
    # process_name_plate(paths["namePlate"])

    print("\nDone. Check outputs inside data/processed/")


if __name__ == "__main__":
    main()