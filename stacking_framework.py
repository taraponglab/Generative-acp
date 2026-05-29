"""
Generic Hold-out Stacking Framework
==================================

Repository-friendly implementation for binary classification tasks.

Inputs:
- Train features CSV
- Train labels CSV
- Test features CSV
- Test labels CSV

Assumptions:
- Labels are stored in a column named "Label"
- Feature matrix can optionally contain a sequence column ("Sequence"/"sequence")
- Features represent flattened sequence encodings with alphabet_size channels
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input,
    LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Add
)

from sklearn.metrics import (
    accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_cnn(input_shape, seed=42):
    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation="relu",
               kernel_initializer=initializer),
        MaxPooling1D(2),
        Flatten(),
        Dense(100, activation="relu",
              kernel_initializer=initializer),
        Dense(1, activation="sigmoid",
              kernel_initializer=initializer)
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def create_transformer(input_shape,
                       embed_dim=128,
                       num_heads=4,
                       ff_dim=128,
                       seed=42):

    initializer = tf.keras.initializers.GlorotUniform(seed=seed)

    inputs = Input(shape=input_shape)

    x = Conv1D(
        embed_dim,
        kernel_size=1,
        activation="relu",
        kernel_initializer=initializer
    )(inputs)

    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        kernel_initializer=initializer
    )(x, x)

    attn = Dropout(0.1, seed=seed)(attn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))

    ffn = Dense(
        ff_dim,
        activation="relu",
        kernel_initializer=initializer
    )(x)

    ffn = Dense(
        embed_dim,
        kernel_initializer=initializer
    )(ffn)

    ffn = Dropout(0.1, seed=seed)(ffn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, ffn]))

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu",
              kernel_initializer=initializer)(x)

    outputs = Dense(
        1,
        activation="sigmoid",
        kernel_initializer=initializer
    )(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def create_meta_model(seed=42):
    return RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1
    )


def evaluate_model(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
    }


def run_stacking(
    X_train,
    y_train,
    X_test,
    y_test,
    output_dir="results",
    n_repeats=3,
    val_size=0.2,
    epochs=30,
    batch_size=32
):

    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for seed in range(n_repeats):

        set_global_seed(seed)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            stratify=y_train,
            random_state=seed
        )

        val_preds = []
        test_preds = []
        metrics = {}

        learners = {
            "CNN": create_cnn,
            "Transformer": create_transformer
        }

        for name, builder in learners.items():

            tf.keras.backend.clear_session()

            model = builder(
                X_train.shape[1:],
                seed=seed
            )

            model.fit(
                X_tr,
                y_tr,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=True
            )

            val_pred = model.predict(X_val, verbose=0).ravel()
            test_pred = model.predict(X_test, verbose=0).ravel()

            val_preds.append(val_pred)
            test_preds.append(test_pred)

            metrics[name] = evaluate_model(
                y_test,
                test_pred
            )

        meta_X_val = np.column_stack(val_preds)
        meta_X_test = np.column_stack(test_preds)

        meta_model = create_meta_model(seed)
        meta_model.fit(meta_X_val, y_val)

        final_prob = meta_model.predict_proba(
            meta_X_test
        )[:, 1]

        metrics["Stacking"] = evaluate_model(
            y_test,
            final_prob
        )

        row = {"Seed": seed}

        for model_name, model_metrics in metrics.items():
            for metric_name, value in model_metrics.items():
                row[f"{model_name}_{metric_name}"] = value

        all_results.append(row)

    raw_df = pd.DataFrame(all_results)
    raw_df.to_csv(
        os.path.join(output_dir, "raw_results.csv"),
        index=False
    )

    summary = []

    for model_name in ["CNN", "Transformer", "Stacking"]:

        row = {"Model": model_name}

        for metric in [
            "Accuracy",
            "MCC",
            "AUROC",
            "AUPRC"
        ]:
            col = f"{model_name}_{metric}"

            row[metric] = (
                f"{raw_df[col].mean():.4f} ± "
                f"{raw_df[col].std():.4f}"
            )

        summary.append(row)

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        os.path.join(output_dir, "summary_results.csv"),
        index=False
    )

    print(summary_df)


def load_features(path):

    df = pd.read_csv(path, index_col=0)

    for col in ["Sequence", "sequence"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = (
        df.apply(pd.to_numeric, errors="coerce")
          .fillna(0)
    )

    return df.values.astype(np.float32)


def reshape_features(X, alphabet_size):

    if X.shape[1] % alphabet_size != 0:
        raise ValueError(
            "Number of features must be divisible "
            "by alphabet_size."
        )

    seq_len = X.shape[1] // alphabet_size

    return X.reshape(
        (-1, seq_len, alphabet_size)
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train-features", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--test-features", required=True)
    parser.add_argument("--test-labels", required=True)

    parser.add_argument(
        "--alphabet-size",
        type=int,
        default=20
    )

    parser.add_argument(
        "--output-dir",
        default="results"
    )

    args = parser.parse_args()

    X_train = load_features(args.train_features)
    X_test = load_features(args.test_features)

    y_train = pd.read_csv(
        args.train_labels
    )["Label"].values

    y_test = pd.read_csv(
        args.test_labels
    )["Label"].values

    X_train = reshape_features(
        X_train,
        args.alphabet_size
    )

    X_test = reshape_features(
        X_test,
        args.alphabet_size
    )

    run_stacking(
        X_train,
        y_train,
        X_test,
        y_test,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
