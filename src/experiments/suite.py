import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier
from src.algorithms.incremental_tree import IncrementalTree
from src.algorithms.incremental_forest import IncrementalForest


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
        self.forest_estimators_comparison(dataset_name, data)
        self.tree_pruning_comparison(dataset_name, data)

    def quality_comparison(self, dataset_name, data, test_size=0.3):
        print("\n>>> [1/5] Quality Comparison (MyTree vs MyForest vs Sklearn)")
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=42
        )
        X_train = train_data[:, :-1].astype(int)
        y_train = train_data[:, -1].astype(int)
        X_test = test_data[:, :-1].astype(int)
        y_test = test_data[:, -1].astype(int)

        # MyTree (Batch)
        t0 = time.time()
        tree_batch = IncrementalTree()
        tree_batch.fit(train_data)
        t_batch = time.time() - t0

        y_pred_batch = self._safe_predict(tree_batch, X_test)
        acc_batch = accuracy_score(y_test, y_pred_batch)
        kappa_batch = cohen_kappa_score(y_test, y_pred_batch)

        # MyTree (Incremental)
        t0 = time.time()
        tree_inc = IncrementalTree()
        for row in train_data:
            tree_inc.update(row)
        t_inc = time.time() - t0

        y_pred_inc = self._safe_predict(tree_inc, X_test)
        acc_inc = accuracy_score(y_test, y_pred_inc)
        kappa_inc = cohen_kappa_score(y_test, y_pred_inc)

        # MyForest (Incremental)
        t0 = time.time()
        forest_inc = IncrementalForest(
            n_estimators=10, min_gain=0.01, random_state=42
        )
        for row in train_data:
            forest_inc.update(row)
        t_forest_inc = time.time() - t0

        y_pred_forest_inc = self._safe_predict_forest(forest_inc, X_test)
        acc_forest_inc = accuracy_score(y_test, y_pred_forest_inc)
        kappa_forest_inc = cohen_kappa_score(y_test, y_pred_forest_inc)

        # Sklearn (CART)
        t0 = time.time()
        clf_sklearn = DecisionTreeClassifier(
            criterion='entropy', random_state=42
        )
        clf_sklearn.fit(X_train, y_train)
        t_sklearn = time.time() - t0

        y_pred_sklearn = clf_sklearn.predict(X_test)
        acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
        kappa_sklearn = cohen_kappa_score(y_test, y_pred_sklearn)

        # Print results
        print(f" - MyTree (Batch):   Acc={acc_batch:.4f}, "
              f"Kappa={kappa_batch:.4f}, Time={t_batch:.4f}s")
        print(f" - MyTree (Inc):     Acc={acc_inc:.4f}, "
              f"Kappa={kappa_inc:.4f}, Time={t_inc:.4f}s")
        print(f" - MyForest (Inc):   Acc={acc_forest_inc:.4f}, "
              f"Kappa={kappa_forest_inc:.4f}, Time={t_forest_inc:.4f}s")
        print(f" - Sklearn (CART):   Acc={acc_sklearn:.4f}, "
              f"Kappa={kappa_sklearn:.4f}, Time={t_sklearn:.4f}s")

        # Print node counts
        node_count_tree = tree_batch.get_node_count()
        node_count_forest = forest_inc.get_node_count()
        avg_nodes = node_count_forest / forest_inc.n_estimators
        print("\n[MEMORY] Node Count:")
        print(f" - MyTree:   {node_count_tree} nodes")
        print(f" - MyForest: {node_count_forest} nodes "
              f"(avg {avg_nodes:.1f} per tree)")

        # Plot quality comparison
        self._plot_quality(
            dataset_name,
            acc_batch, acc_inc, acc_forest_inc, acc_sklearn,
            t_batch, t_inc, t_forest_inc, t_sklearn
        )

        # Generate and save confusion matrix heatmap for Forest
        self._plot_confusion_matrix(
            dataset_name,
            y_test,
            y_pred_forest_inc,
            "MyForest_Inc"
        )

    def time_performance(self, dataset_name, data):
        print("\n>>> [2/5] Time Performance (Stream Simulation)")
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
                      f"{t_retrain:.6f}   | {speedup:.1f}x")

                n_samples.append(i)
                times_update.append(t_update)
                times_retrain.append(t_retrain)
                speedups.append(speedup)

        self._plot_time_performance(
            dataset_name, n_samples, times_update, times_retrain, speedups
        )

    def structure_stability(self, dataset_name, data, iterations=3):
        print("\n>>> [3/5] Structure Stability")
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

    def forest_estimators_comparison(self, dataset_name, data, test_size=0.3):
        """Compare forests with different numbers of estimators."""
        print("\n>>> [4/5] Forest Estimators Comparison (Incremental)")
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=42
        )
        X_test = test_data[:, :-1].astype(int)
        y_test = test_data[:, -1].astype(int)

        estimator_counts = [5, 10, 20, 50]
        results = []

        for n_est in estimator_counts:
            forest = IncrementalForest(
                n_estimators=n_est, min_gain=0.01, random_state=42
            )
            # Train incrementally
            for row in train_data:
                forest.update(row)

            y_pred = self._safe_predict_forest(forest, X_test)
            acc = accuracy_score(y_test, y_pred)
            node_count = forest.get_node_count()
            results.append({
                'n_estimators': n_est,
                'accuracy': acc,
                'node_count': node_count
            })
            print(f" - Forest(n={n_est}): Acc={acc:.4f}, Nodes={node_count}")

        self._plot_forest_estimators(dataset_name, results)

    def tree_pruning_comparison(self, dataset_name, data, test_size=0.3):
        """Compare trees with different pruning (min_gain) values."""
        print("\n>>> [5/5] Tree Pruning Comparison (Incremental)")
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=42
        )
        X_test = test_data[:, :-1].astype(int)
        y_test = test_data[:, -1].astype(int)

        min_gain_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        results = []

        for mg in min_gain_values:
            tree = IncrementalTree(min_gain=mg)
            # Train incrementally
            for row in train_data:
                tree.update(row)

            y_pred = self._safe_predict(tree, X_test)
            acc = accuracy_score(y_test, y_pred)
            node_count = tree.get_node_count()
            results.append({
                'min_gain': mg,
                'accuracy': acc,
                'node_count': node_count
            })
            print(f" - Tree(min_gain={mg}): Acc={acc:.4f}, Nodes={node_count}")

        self._plot_tree_pruning(dataset_name, results)

    def _safe_predict(self, tree, X):
        raw_preds = [tree.predict(row) for row in X]
        return [p if p is not None else -1 for p in raw_preds]

    def _safe_predict_forest(self, forest, X):
        raw_preds = [forest.predict(row) for row in X]
        return [p if p is not None else -1 for p in raw_preds]

    def _plot_quality(self, name, acc_batch, acc_inc, acc_forest, acc_sk,
                      t_batch, t_inc, t_forest, t_sk):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        models = ['Tree(Batch)', 'Tree(Inc)', 'Forest(Inc)', 'Sklearn']
        accs = [acc_batch, acc_inc, acc_forest, acc_sk]
        colors = ['skyblue', 'lightgreen', 'coral', 'lightgray']

        bars = ax1.bar(models, accs, color=colors)
        ax1.set_title(f'{name} - Accuracy Comparison')
        ax1.set_ylim(0, 1.1)
        ax1.set_ylabel('Accuracy')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom')

        times = [t_batch, t_inc, t_forest, t_sk]
        bar_colors = ['salmon', 'orange', 'purple', 'gray']
        bars2 = ax2.bar(models, times, color=bar_colors)
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

    def _plot_confusion_matrix(self, name, y_true, y_pred, model_name):
        """Generate and save a confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f'{name} - Confusion Matrix ({model_name})')

        filename = f'{name}_confusion_matrix_{model_name}.png'
        path = os.path.join(self.results_dir, filename)
        plt.savefig(path)
        plt.close()
        print(f"[PLOT] Saved: {path}")

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

        ax2.plot(
            n, speedups, label='Speedup Factor', color='green', marker='^'
        )
        ax2.set_xlabel('Number of Samples')
        ax2.set_ylabel('Speedup (x times)')
        ax2.set_title(f'{name} - Incremental Speedup')
        ax2.grid(True)

        path = os.path.join(self.results_dir, f'{name}_performance.png')
        plt.savefig(path)
        plt.close()
        print(f"[PLOT] Saved: {path}")

    def _plot_forest_estimators(self, name, results):
        """Plot forest comparison with different n_estimators."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        n_ests = [r['n_estimators'] for r in results]
        accs = [r['accuracy'] for r in results]
        nodes = [r['node_count'] for r in results]
        labels = [f'n={n}' for n in n_ests]
        colors = ['#2ecc71', '#3498db', '#9b59b6']

        # Accuracy plot
        bars = ax1.bar(labels, accs, color=colors)
        ax1.set_title(f'{name} - Forest Accuracy by n_estimators (Inc)')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        ax1.set_xlabel('Number of Estimators')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom')

        # Node count plot
        bars2 = ax2.bar(labels, nodes, color=colors)
        ax2.set_title(f'{name} - Forest Node Count by n_estimators (Inc)')
        ax2.set_ylabel('Total Nodes')
        ax2.set_xlabel('Number of Estimators')
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        path = os.path.join(self.results_dir, f'{name}_forest_estimators.png')
        plt.savefig(path)
        plt.close()
        print(f"[PLOT] Saved: {path}")

    def _plot_tree_pruning(self, name, results):
        """Plot tree comparison with different min_gain (pruning) values."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        min_gains = [r['min_gain'] for r in results]
        accs = [r['accuracy'] for r in results]
        nodes = [r['node_count'] for r in results]
        labels = [f'mg={mg}' for mg in min_gains]
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#1abc9c', '#3498db']

        # Accuracy plot
        bars = ax1.bar(labels, accs, color=colors)
        ax1.set_title(f'{name} - Tree Accuracy by min_gain (Inc)')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        ax1.set_xlabel('Pruning Threshold (min_gain)')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom')

        # Node count plot
        bars2 = ax2.bar(labels, nodes, color=colors)
        ax2.set_title(f'{name} - Tree Node Count by min_gain (Inc)')
        ax2.set_ylabel('Total Nodes')
        ax2.set_xlabel('Pruning Threshold (min_gain)')
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        path = os.path.join(self.results_dir, f'{name}_tree_pruning.png')
        plt.savefig(path)
        plt.close()
        print(f"[PLOT] Saved: {path}")
