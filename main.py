import os
import traceback
from src.utils.data_loader import DataLoader
from src.experiments.suite import ExperimentSuite


def main():
    if not os.path.exists('data'):
        os.makedirs('data')

    loader = DataLoader(bins=5)
    suite = ExperimentSuite()

    datasets_to_run = [
        ('IRIS', lambda: loader.load_iris()),
        ('WEATHER', lambda: loader.load_australian_weather()),
        ('AIRLINES', lambda: loader.load_airlines())
    ]

    for name, load_func in datasets_to_run:
        try:
            data = load_func()
            suite.run_all(name, data)
        except Exception as e:
            print(f"CRITICAL ERROR IN {name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
