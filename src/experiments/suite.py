import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from src.algorithms.incremental_tree import IncrementalTree


class ExperimentSuite:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        print(f"[INFO] Results saved to: {os.path.abspath(self.results_dir)}")

    def run_all(self, dataset_name, data):
        if data is None:
            print(f"SKIPPED: No data for {dataset_name}")
            return

        print(f"\n{'='*60}")
        print(f" START EXPERIMENTS: {dataset_name.upper()}")
        print(f" Samples: {len(data)}, Features: {data.shape[1]-1}")
        print(f"{'='*60}")

        self.quality_comparison(dataset_name, data)
        self.time_performance(dataset_name, data)
        self.structure_stability(dataset_name, data)

    def quality_comparison(self, dataset_name, data, test_size=0.3):
        print("\n>>> [1/3] Quality Comparison (MyTree vs Sklearn)")
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=42
        )
        X_train = train_data[:, :-1].astype(int)
        y_train = train_data[:, -1].astype(int)
        X_test = test_data[:, :-1].astype(int)
        y_test = test_data[:, -1].astype(int)

        t0 = time.time()
        tree_batch = IncrementalTree()
        tree_batch.fit(train_data)
        t_batch = time.time() - t0

        y_pred_batch = self._safe_predict(tree_batch, X_test)
        acc_batch = accuracy_score(y_test, y_pred_batch)

        t0 = time.time()
        tree_inc = IncrementalTree()
        for row in train_data:
            tree_inc.update(row)
        t_inc = time.time() - t0

        y_pred_inc = self._safe_predict(tree_inc, X_test)
        acc_inc = accuracy_score(y_test, y_pred_inc)

        t0 = time.time()
        clf_sklearn = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf_sklearn.fit(X_train, y_train)
        t_sklearn = time.time() - t0

        y_pred_sklearn = clf_sklearn.predict(X_test)
        acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

        print(f" - MyTree (Batch): Acc={acc_batch:.4f}, Time={t_batch:.4f}s")
        print(f" - MyTree (Inc):   Acc={acc_inc:.4f},   Time={t_inc:.4f}s")
        print(f" - Sklearn (CART): Acc={acc_sklearn:.4f}, Time={t_sklearn:.4f}s")

        self._plot_quality(
            dataset_name,
            acc_batch, acc_inc, acc_sklearn,
            t_batch, t_inc, t_sklearn
        )

    def time_performance(self, dataset_name, data):
        print("\n>>> [2/3] Time Performance (Stream Simulation)")
        limit = min(len(data), 2000)
        subset = data[:limit]

        tree_inc = IncrementalTree()
        tree_batch = IncrementalTree()
        history = []

        n_samples = []
        times_update = []
        times_retrain = []
        speedups = []

        print(f"{'N':<5} | {'Update(s)':<10} | "
              f"{'Retrain(s)':<10} | {'Speedup':<8}")
        print("-" * 45)

        report_interval = max(10, limit // 20)

        for i, row in enumerate(subset):
            history.append(row)

            t0 = time.time()
            tree_inc.update(row)
            t_update = time.time() - t0

            if i > 0 and i % report_interval == 0:
                t0 = time.time()
                tree_batch.fit(np.array(history))
                t_retrain = time.time() - t0

                speedup = t_retrain / t_update if t_update > 1e-9 else 0

                print(f"{i:<5} | {t_update:.6f}   | "
                      f"{t_retrain:.6f}    | {speedup:.1f}x")

                n_samples.append(i)
                times_update.append(t_update)
                times_retrain.append(t_retrain)
                speedups.append(speedup)

        self._plot_time_performance(
            dataset_name, n_samples, times_update, times_retrain, speedups
        )

    def structure_stability(self, dataset_name, data, iterations=3):
        print("\n>>> [3/3] Structure Stability")
        limit = min(len(data), 1000)
        subset = data[:limit]

        roots = []
        for _ in range(iterations):
            np.random.shuffle(subset)
            tree = IncrementalTree()
            for row in subset:
                tree.update(row)

            if not tree.root.is_leaf():
                feat = tree.root.feature
            else:
                feat = "Leaf"
            roots.append(feat)

        unique = set(roots)
        if len(unique) == 1:
            print(f" - STABLE. Root feature: {list(unique)[0]}")
        else:
            print(f" - UNSTABLE. Root features: {unique}")

    def _safe_predict(self, tree, X):
        raw_preds = [tree.predict(row) for row in X]
        return [p if p is not None else -1 for p in raw_preds]

    def _plot_quality(self, name, acc_batch, acc_inc, acc_sk, t_batch, t_inc, t_sk):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        models = ['MyTree (Batch)', 'MyTree (Inc)', 'Sklearn']
        accs = [acc_batch, acc_inc, acc_sk]
        colors = ['skyblue', 'lightgreen', 'lightgray']

        bars = ax1.bar(models, accs, color=colors)
        ax1.set_title(f'{name} - Accuracy Comparison')
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel('Accuracy')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom')

        times = [t_batch, t_inc, t_sk]
        bars2 = ax2.bar(models, times, color=['salmon', 'orange', 'gray'])
        ax2.set_title(f'{name} - Training Time (Log Scale)')
        ax2.set_ylabel('Time (s)')
        ax2.set_yscale('log')
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom')

        path = os.path.join(self.results_dir, f'{name}_quality_vs_sklearn.png')
        plt.savefig(path)
        plt.close()
        print(f"[PLOT] Saved: {path}")

    def _plot_time_performance(self, name, n, t_up, t_ret, speedups):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(n, t_ret, label='Retrain (MyBatch)', marker='o')
        ax1.plot(n, t_up, label='Update (MyInc)', marker='x')
        ax1.set_xlabel('Number of Samples')
        ax1.set_ylabel('Time (s)')
        ax1.set_title(f'{name} - Update vs Retrain Time')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(n, speedups, label='Speedup Factor', color='green', marker='^')
        ax2.set_xlabel('Number of Samples')
        ax2.set_ylabel('Speedup (x times)')
        ax2.set_title(f'{name} - Incremental Speedup')
        ax2.grid(True)

        path = os.path.join(self.results_dir, f'{name}_performance.png')
        plt.savefig(path)
        plt.close()
        print(f"[PLOT] Saved: {path}")
