# ============================================================
# MALWARE IMAGE CLASSIFICATION USING CNN
# 10-FOLD STRATIFIED CROSS VALIDATION
#
# Metrics:
#   - Accuracy
#   - Precision
#   - Recall
#   - Macro F1
#   - Weighted F1
#   - Confusion Matrix
#   - Classification Report
#   - Parameters
#   - FLOPs
#   - MACs
#   - Model Size
#   - Inference Time
#   - Latency / Image
#   - Throughput
#   - Mean +/- Std across 10 folds
#
# ============================================================

import os
import glob
import csv
import numpy as np
import cv2
import time
import random
import warnings

import tensorflow as tf
import keras

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Activation
)

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import losses
from keras import regularizers


# ============================================================
# SETTINGS
# ============================================================

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Dataset path
DATASET_PATH = r'D:\Malware Paper\Malware\Microsoft_Image_Dataset'

# Model output path
MODEL_OUTPUT_PATH = (
    r'D:\Malware Paper\Malware\trained_model'
)

# Number of folds
KFOLD = 10

# Number of epochs
EPOCHS = 20

# Batch size
BATCH_SIZE = 1

# Input image size
IMAGE_SIZE = 64

# Random seed
RANDOM_SEED = 42


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# TRANSITION MATRIX
# ============================================================

def transition_matrix(arr, n=1):

    """
    Computes the transition matrix from Markov chain
    sequence of order n.
    """

    M = np.zeros(
        shape=(max(arr) + 1, max(arr) + 1)
    )

    for (i, j) in zip(arr, arr[1:]):
        M[i, j] += 1

    M_sum = np.ma.masked_equal(
        M.sum(axis=1),
        0
    )

    T = (M.T / M_sum).T

    return np.linalg.matrix_power(T, n)


# ============================================================
# GET NUMBER OF IMAGES PER FAMILY
# ============================================================

def num_img_fams_list():

    print("\n")
    print("=" * 80)
    print("READING DATASET")
    print("=" * 80)

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Dataset path does not exist:\n{DATASET_PATH}"
        )

    # Parent directory
    os.chdir(DATASET_PATH)

    # Get only directories
    list_fams = [
        folder
        for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(
            os.path.join(DATASET_PATH, folder)
        )
    ]

    # Sort for reproducibility
    list_fams.sort()

    no_imgs = []

    for family in list_fams:

        family_path = os.path.join(
            DATASET_PATH,
            family
        )

        no_per_family = len(
            glob.glob(
                os.path.join(
                    family_path,
                    '*.png'
                )
            )
        )

        no_imgs.append(no_per_family)

    print("\nNumber of malware families:")
    print(len(list_fams))

    print("\nFamilies:")

    for i, family in enumerate(list_fams):

        print(
            f"{i:3d} : "
            f"{family:<40} "
            f"{no_imgs[i]}"
        )

    return no_imgs, list_fams


# ============================================================
# GENERATE LABELS
# ============================================================

def getlabels(no_imgs, total):

    y = np.zeros(
        total,
        dtype=np.int32
    )

    temp1 = np.zeros(
        len(no_imgs) + 1
    )

    temp1[1:] = no_imgs

    temp2 = int(temp1[0])

    for label in range(len(no_imgs)):

        temp3 = (
            temp2 +
            int(temp1[label + 1])
        )

        for index in range(
            temp2,
            temp3
        ):

            y[index] = label

        temp2 = (
            temp2 +
            int(temp1[label + 1])
        )

    return y


# ============================================================
# LOAD AND PREPROCESS IMAGES
# ============================================================

def getX(list_fams, total):

    trans_mat_arr = []

    total_time = 0.0

    min_time = float('inf')

    max_time = 0.0

    image_counter = 0

    print("\n")
    print("=" * 80)
    print("PREPROCESSING IMAGES")
    print("=" * 80)

    for family_index, family in enumerate(list_fams):

        family_path = os.path.join(
            DATASET_PATH,
            family
        )

        img_list = glob.glob(
            os.path.join(
                family_path,
                '*.png'
            )
        )

        print(
            f"Processing "
            f"{family_index + 1}/{len(list_fams)} : "
            f"{family} "
            f"({len(img_list)} images)"
        )

        for img_path in img_list:

            img = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:

                print(
                    f"WARNING: Could not read {img_path}"
                )

                continue

            tic = time.perf_counter()

            # Histogram equalization
            cl1 = cv2.equalizeHist(img)

            # Resize to 64 x 64
            cl1 = cv2.resize(
                cl1,
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                interpolation=cv2.INTER_AREA
            )

            toc = time.perf_counter()

            processing_time = toc - tic

            total_time += processing_time

            min_time = min(
                min_time,
                processing_time
            )

            max_time = max(
                max_time,
                processing_time
            )

            trans_mat_arr.append(cl1)

            image_counter += 1

    if image_counter == 0:

        raise RuntimeError(
            "No images were loaded from the dataset."
        )

    print("\nPreprocessing statistics:")

    print(
        f"Images processed       : "
        f"{image_counter}"
    )

    print(
        f"Average time/image     : "
        f"{total_time / image_counter:.8f} seconds"
    )

    print(
        f"Minimum time/image     : "
        f"{min_time:.8f} seconds"
    )

    print(
        f"Maximum time/image     : "
        f"{max_time:.8f} seconds"
    )

    return np.array(
        trans_mat_arr,
        dtype=np.uint8
    )


# ============================================================
# STRATIFIED K-FOLD
# ============================================================

def strtifiedkfolddata(
    X,
    y,
    total,
    kfold
):

    skf = StratifiedKFold(
        n_splits=kfold,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    skfind = []

    for train_index, test_index in skf.split(
        X,
        y
    ):

        skfind.append(
            (
                train_index,
                test_index
            )
        )

    return skfind


# ============================================================
# CNN MODEL
# ============================================================

def model_cnn(no_imgs):

    model = Sequential()

    # --------------------------------------------------------
    # Conv Block 1
    # --------------------------------------------------------

    model.add(
        Conv2D(
            64,
            (2, 2),
            padding='same',
            input_shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
                1
            ),
            activation='relu',
            kernel_regularizer=regularizers.l2(
                0.001
            )
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )
    )

    # --------------------------------------------------------
    # Conv Block 2
    # --------------------------------------------------------

    model.add(
        Conv2D(
            128,
            (2, 2),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(
                0.001
            )
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )
    )

    # --------------------------------------------------------
    # Conv Block 3
    # --------------------------------------------------------

    model.add(
        Conv2D(
            256,
            (2, 2),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(
                0.001
            )
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )
    )

    # --------------------------------------------------------
    # Conv Block 4
    # --------------------------------------------------------

    model.add(
        Conv2D(
            512,
            (2, 2),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(
                0.001
            )
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )
    )

    # --------------------------------------------------------
    # Conv Block 5
    # --------------------------------------------------------

    model.add(
        Conv2D(
            512,
            (2, 2),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(
                0.001
            )
        )
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )
    )

    # --------------------------------------------------------
    # Fully Connected Layers
    # --------------------------------------------------------

    model.add(
        Dropout(0.00)
    )

    model.add(
        Flatten()
    )

    model.add(
        Dense(512)
    )

    model.add(
        Dense(512)
    )

    model.add(
        Dense(256)
    )

    model.add(
        Dense(128)
    )

    model.add(
        Dense(len(no_imgs))
    )

    model.add(
        Activation('softmax')
    )

    return model


# ============================================================
# COUNT PARAMETERS
# ============================================================

def get_parameter_statistics(model):

    total_params = model.count_params()

    trainable_params = sum(
        np.prod(v.shape)
        for v in model.trainable_variables
    )

    non_trainable_params = sum(
        np.prod(v.shape)
        for v in model.non_trainable_variables
    )

    return (
        int(total_params),
        int(trainable_params),
        int(non_trainable_params)
    )


# ============================================================
# CALCULATE FLOPs
# ============================================================

def get_flops(model):

    """
    Calculates FLOPs for a single input image.

    Input:
        1 x 64 x 64 x 1

    Returns:
        Total FLOPs
    """

    try:

        input_signature = [
            tf.TensorSpec(
                shape=(
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    1
                ),
                dtype=tf.float32
            )
        ]

        @tf.function(
            input_signature=input_signature
        )
        def forward(x):

            return model(x)

        concrete_func = forward.get_concrete_function()

        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph
        )

        frozen_func, graph_def = (
            convert_variables_to_constants_v2_as_graph(
                concrete_func
            )
        )

        with tf.Graph().as_default() as graph:

            tf.graph_util.import_graph_def(
                graph_def,
                name=''
            )

            run_meta = tf.compat.v1.RunMetadata()

            opts = (
                tf.compat.v1.profiler
                .ProfileOptionBuilder
                .float_operation()
            )

            opts['output'] = 'none'

            flops = tf.compat.v1.profiler.profile(
                graph=graph,
                run_meta=run_meta,
                cmd='op',
                options=opts
            )

        if flops is not None:

            return int(
                flops.total_float_ops
            )

        return None

    except Exception as e:

        print(
            "\nWARNING: FLOPs calculation failed."
        )

        print(
            str(e)
        )

        return None


# ============================================================
# MODEL COMPLEXITY
# ============================================================

def calculate_model_complexity(model):

    (
        total_params,
        trainable_params,
        non_trainable_params
    ) = get_parameter_statistics(model)

    print("\n")
    print("=" * 80)
    print("MODEL COMPLEXITY")
    print("=" * 80)

    print(
        f"Total parameters       : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters   : "
        f"{trainable_params:,}"
    )

    print(
        f"Non-trainable params   : "
        f"{non_trainable_params:,}"
    )

    # --------------------------------------------------------
    # Model size assuming float32
    # --------------------------------------------------------

    model_size_bytes = (
        total_params * 4
    )

    model_size_mb = (
        model_size_bytes /
        (1024 ** 2)
    )

    print(
        f"Approx. model size     : "
        f"{model_size_mb:.3f} MB"
    )

    # --------------------------------------------------------
    # FLOPs
    # --------------------------------------------------------

    flops = get_flops(model)

    if flops is not None:

        macs = flops / 2.0

        print(
            f"FLOPs                   : "
            f"{flops:,}"
        )

        print(
            f"FLOPs                   : "
            f"{flops / 1e6:.4f} MFLOPs"
        )

        print(
            f"FLOPs                   : "
            f"{flops / 1e9:.4f} GFLOPs"
        )

        print(
            f"MACs                    : "
            f"{macs:,.0f}"
        )

        print(
            f"MACs                    : "
            f"{macs / 1e6:.4f} MMACs"
        )

    else:

        macs = None

        print(
            "FLOPs                   : N/A"
        )

        print(
            "MACs                    : N/A"
        )

    print("=" * 80)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'model_size_mb': model_size_mb,
        'flops': flops,
        'macs': macs
    }


# ============================================================
# INFERENCE PERFORMANCE
# ============================================================

def measure_inference_time(
    model,
    X_test
):

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warmup_size = min(
        10,
        len(X_test)
    )

    model.predict(
        X_test[:warmup_size],
        verbose=0
    )

    # --------------------------------------------------------
    # Actual inference
    # --------------------------------------------------------

    tic = time.perf_counter()

    predictions = model.predict(
        X_test,
        verbose=0
    )

    toc = time.perf_counter()

    total_time = toc - tic

    num_images = len(X_test)

    latency_per_image = (
        total_time /
        num_images
    )

    throughput = (
        num_images /
        total_time
    )

    print("\n")
    print("=" * 80)
    print("INFERENCE PERFORMANCE")
    print("=" * 80)

    print(
        f"Test images            : "
        f"{num_images}"
    )

    print(
        f"Total inference time   : "
        f"{total_time:.6f} seconds"
    )

    print(
        f"Average latency/image  : "
        f"{latency_per_image * 1000:.4f} ms"
    )

    print(
        f"Throughput             : "
        f"{throughput:.2f} images/sec"
    )

    print("=" * 80)

    return (
        predictions,
        total_time,
        latency_per_image,
        throughput
    )


# ============================================================
# SAVE CSV RESULTS
# ============================================================

def save_results_csv(
    fold_results,
    complexity,
    output_file
):

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    fieldnames = [
        'Fold',
        'Accuracy',
        'Precision_Macro',
        'Recall_Macro',
        'F1_Macro',
        'F1_Weighted',
        'Inference_Time_sec',
        'Latency_ms',
        'Throughput_images_sec'
    ]

    with open(
        output_file,
        'w',
        newline=''
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in fold_results:

            writer.writerow(row)

    print(
        f"\nFold results saved to:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def train():

    # ========================================================
    # LOAD DATASET
    # ========================================================

    no_imgs, list_fams = (
        num_img_fams_list()
    )

    total = sum(no_imgs)

    print("\n")
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(
        f"Total images            : "
        f"{total}"
    )

    print(
        f"Total families          : "
        f"{len(list_fams)}"
    )

    # ========================================================
    # LOAD IMAGES
    # ========================================================

    X = getX(
        list_fams,
        total
    )

    # ========================================================
    # LABELS
    # ========================================================

    y = getlabels(
        no_imgs,
        len(X)
    )

    # ========================================================
    # SHUFFLE
    # ========================================================

    p = np.arange(
        len(X)
    )

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(p)

    X = X[p]

    y = y[p]

    # ========================================================
    # STRATIFIED K-FOLD
    # ========================================================

    skfind = strtifiedkfolddata(
        X,
        y,
        len(X),
        KFOLD
    )

    # ========================================================
    # METRIC STORAGE
    # ========================================================

    fold_accuracy = []

    fold_precision = []

    fold_recall = []

    fold_f1_macro = []

    fold_f1_weighted = []

    fold_inference_time = []

    fold_latency = []

    fold_throughput = []

    all_confusion_matrices = []

    fold_results = []

    complexity = None

    # ========================================================
    # K-FOLD LOOP
    # ========================================================

    for i in range(KFOLD):

        print("\n\n")

        print(
            "*" * 90
        )

        print(
            f"FOLD {i + 1}/{KFOLD}"
        )

        print(
            "*" * 90
        )

        # ----------------------------------------------------
        # TRAIN / TEST INDICES
        # ----------------------------------------------------

        train_indices = skfind[i][0]

        test_indices = skfind[i][1]

        # ----------------------------------------------------
        # TRAINING DATA
        # ----------------------------------------------------

        X_train = X[
            train_indices
        ]

        X_train = (
            X_train
            .reshape(
                -1,
                IMAGE_SIZE,
                IMAGE_SIZE,
                1
            )
            .astype(
                np.float32
            )
        )

        y_train = y[
            train_indices
        ]

        y_train = keras.utils.to_categorical(
            y_train,
            num_classes=len(no_imgs)
        )

        # ----------------------------------------------------
        # TEST DATA
        # ----------------------------------------------------

        X_test = X[
            test_indices
        ]

        X_test = (
            X_test
            .reshape(
                -1,
                IMAGE_SIZE,
                IMAGE_SIZE,
                1
            )
            .astype(
                np.float32
            )
        )

        y_test = y[
            test_indices
        ].astype(
            np.int32
        )

        y_test_onehot = (
            keras.utils.to_categorical(
                y_test,
                num_classes=len(no_imgs)
            )
        )

        print(
            f"Training samples       : "
            f"{len(X_train)}"
        )

        print(
            f"Testing samples        : "
            f"{len(X_test)}"
        )

        # ====================================================
        # CREATE MODEL
        # ====================================================

        model = model_cnn(
            no_imgs
        )

        # ====================================================
        # MODEL COMPLEXITY
        # ====================================================

        if i == 0:

            print("\n")
            print(
                "=" * 80
            )

            print(
                "CNN ARCHITECTURE"
            )

            print(
                "=" * 80
            )

            model.summary()

            complexity = (
                calculate_model_complexity(
                    model
                )
            )

        # ====================================================
        # OPTIMIZER
        # ====================================================

        opt = SGD(
            learning_rate=0.001,
            momentum=0.2
        )

        # ====================================================
        # COMPILE
        # ====================================================

        model.compile(
            optimizer=opt,
            loss=losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # ====================================================
        # TRAIN
        # ====================================================

        print("\n")
        print(
            "=" * 80
        )

        print(
            f"TRAINING FOLD {i + 1}"
        )

        print(
            "=" * 80
        )

        training_start = time.perf_counter()

        history = model.fit(
            X_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=1
        )

        training_end = time.perf_counter()

        training_time = (
            training_end -
            training_start
        )

        print(
            f"\nTraining time: "
            f"{training_time:.4f} seconds"
        )

        # ====================================================
        # MODEL EVALUATION
        # ====================================================

        scores = model.evaluate(
            X_test,
            y_test_onehot,
            verbose=0
        )

        keras_accuracy = scores[1]

        print(
            f"\nKeras test accuracy: "
            f"{keras_accuracy * 100:.4f}%"
        )

        # ====================================================
        # INFERENCE
        # ====================================================

        (
            y_predict_probability,
            inference_time,
            latency_per_image,
            throughput
        ) = measure_inference_time(
            model,
            X_test
        )

        # ====================================================
        # PREDICTED LABELS
        # ====================================================

        y_predict = np.argmax(
            y_predict_probability,
            axis=1
        )

        # ====================================================
        # CLASSIFICATION METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_test,
            y_predict
        )

        precision = precision_score(
            y_test,
            y_predict,
            average='macro',
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_predict,
            average='macro',
            zero_division=0
        )

        f1_macro = f1_score(
            y_test,
            y_predict,
            average='macro',
            zero_division=0
        )

        f1_weighted = f1_score(
            y_test,
            y_predict,
            average='weighted',
            zero_division=0
        )

        # ====================================================
        # STORE METRICS
        # ====================================================

        fold_accuracy.append(
            accuracy * 100
        )

        fold_precision.append(
            precision * 100
        )

        fold_recall.append(
            recall * 100
        )

        fold_f1_macro.append(
            f1_macro * 100
        )

        fold_f1_weighted.append(
            f1_weighted * 100
        )

        fold_inference_time.append(
            inference_time
        )

        fold_latency.append(
            latency_per_image * 1000
        )

        fold_throughput.append(
            throughput
        )

        # ====================================================
        # FOLD RESULTS
        # ====================================================

        fold_results.append(
            {
                'Fold': i + 1,
                'Accuracy': accuracy * 100,
                'Precision_Macro': precision * 100,
                'Recall_Macro': recall * 100,
                'F1_Macro': f1_macro * 100,
                'F1_Weighted': f1_weighted * 100,
                'Inference_Time_sec': inference_time,
                'Latency_ms': latency_per_image * 1000,
                'Throughput_images_sec': throughput
            }
        )

        # ====================================================
        # PRINT FOLD RESULTS
        # ====================================================

        print("\n")
        print(
            "=" * 80
        )

        print(
            f"RESULTS - FOLD {i + 1}"
        )

        print(
            "=" * 80
        )

        print(
            f"Accuracy              : "
            f"{accuracy * 100:.4f}%"
        )

        print(
            f"Macro Precision       : "
            f"{precision * 100:.4f}%"
        )

        print(
            f"Macro Recall          : "
            f"{recall * 100:.4f}%"
        )

        print(
            f"Macro F1              : "
            f"{f1_macro * 100:.4f}%"
        )

        print(
            f"Weighted F1           : "
            f"{f1_weighted * 100:.4f}%"
        )

        print(
            f"Inference time        : "
            f"{inference_time:.6f} sec"
        )

        print(
            f"Latency/image         : "
            f"{latency_per_image * 1000:.4f} ms"
        )

        print(
            f"Throughput            : "
            f"{throughput:.2f} images/sec"
        )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        cm = confusion_matrix(
            y_test,
            y_predict,
            labels=np.arange(
                len(list_fams)
            )
        )

        all_confusion_matrices.append(
            cm
        )

        print("\n")
        print(
            "CONFUSION MATRIX"
        )

        print(cm)

        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        print("\n")
        print(
            "CLASSIFICATION REPORT"
        )

        print(
            classification_report(
                y_test,
                y_predict,
                labels=np.arange(
                    len(list_fams)
                ),
                target_names=list_fams,
                zero_division=0,
                digits=4
            )
        )

        # ====================================================
        # SAVE MODEL
        # ====================================================

        avg_acc = np.mean(
            fold_accuracy
        )

        save_path = os.path.join(
            MODEL_OUTPUT_PATH,
            f'models_fold_{i + 1}'
        )

        os.makedirs(
            save_path,
            exist_ok=True
        )

        model_file = os.path.join(
            save_path,
            f'model_acc_{avg_acc:.4f}.h5'
        )

        model.save(
            model_file
        )

        print(
            f"\nModel saved to:\n"
            f"{model_file}"
        )

        # ====================================================
        # CLEAR SESSION
        # ====================================================

        del model

        tf.keras.backend.clear_session()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n\n")

    print(
        "#" * 100
    )

    print(
        "FINAL 10-FOLD CROSS-VALIDATION RESULTS"
    )

    print(
        "#" * 100
    )

    # ========================================================
    # HELPER FUNCTION
    # ========================================================

    def print_mean_std(
        name,
        values,
        unit='%'
    ):

        mean = np.mean(values)

        std = np.std(values)

        if unit == '%':

            print(
                f"{name:<30}: "
                f"{mean:.4f} ± {std:.4f} %"
            )

        elif unit == 'ms':

            print(
                f"{name:<30}: "
                f"{mean:.4f} ± {std:.4f} ms"
            )

        elif unit == 'sec':

            print(
                f"{name:<30}: "
                f"{mean:.6f} ± {std:.6f} sec"
            )

        else:

            print(
                f"{name:<30}: "
                f"{mean:.4f} ± {std:.4f}"
            )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    print("\n")
    print(
        "CLASSIFICATION PERFORMANCE"
    )

    print(
        "-" * 80
    )

    print_mean_std(
        "Accuracy",
        fold_accuracy
    )

    print_mean_std(
        "Macro Precision",
        fold_precision
    )

    print_mean_std(
        "Macro Recall",
        fold_recall
    )

    print_mean_std(
        "Macro F1",
        fold_f1_macro
    )

    print_mean_std(
        "Weighted F1",
        fold_f1_weighted
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    print("\n")
    print(
        "INFERENCE PERFORMANCE"
    )

    print(
        "-" * 80
    )

    print_mean_std(
        "Inference Time",
        fold_inference_time,
        'sec'
    )

    print_mean_std(
        "Latency",
        fold_latency,
        'ms'
    )

    print_mean_std(
        "Throughput",
        fold_throughput,
        'images/sec'
    )

    # ========================================================
    # MODEL COMPLEXITY
    # ========================================================

    print("\n")
    print(
        "MODEL COMPLEXITY"
    )

    print(
        "-" * 80
    )

    if complexity is not None:

        print(
            f"{'Total Parameters':<30}: "
            f"{complexity['total_params']:,}"
        )

        print(
            f"{'Trainable Parameters':<30}: "
            f"{complexity['trainable_params']:,}"
        )

        print(
            f"{'Non-trainable Parameters':<30}: "
            f"{complexity['non_trainable_params']:,}"
        )

        print(
            f"{'Model Size':<30}: "
            f"{complexity['model_size_mb']:.4f} MB"
        )

        if complexity['flops'] is not None:

            print(
                f"{'FLOPs':<30}: "
                f"{complexity['flops']:,}"
            )

            print(
                f"{'GFLOPs':<30}: "
                f"{complexity['flops'] / 1e9:.6f}"
            )

            print(
                f"{'MACs':<30}: "
                f"{complexity['macs']:,.0f}"
            )

            print(
                f"{'MMACs':<30}: "
                f"{complexity['macs'] / 1e6:.6f}"
            )

    # ========================================================
    # FOLD-WISE TABLE
    # ========================================================

    print("\n")
    print(
        "FOLD-WISE PERFORMANCE"
    )

    print(
        "-" * 110
    )

    print(
        f"{'Fold':<8}"
        f"{'Accuracy':<15}"
        f"{'Precision':<15}"
        f"{'Recall':<15}"
        f"{'Macro F1':<15}"
        f"{'Weighted F1':<15}"
        f"{'Latency(ms)':<15}"
    )

    print(
        "-" * 110
    )

    for i in range(KFOLD):

        print(
            f"{i + 1:<8}"
            f"{fold_accuracy[i]:<15.4f}"
            f"{fold_precision[i]:<15.4f}"
            f"{fold_recall[i]:<15.4f}"
            f"{fold_f1_macro[i]:<15.4f}"
            f"{fold_f1_weighted[i]:<15.4f}"
            f"{fold_latency[i]:<15.4f}"
        )

    print(
        "-" * 110
    )

    # ========================================================
    # AGGREGATED CONFUSION MATRIX
    # ========================================================

    overall_cm = np.sum(
        all_confusion_matrices,
        axis=0
    )

    print("\n")
    print(
        "=" * 80
    )

    print(
        "AGGREGATED CONFUSION MATRIX"
    )

    print(
        "=" * 80
    )

    print(
        overall_cm
    )

    # ========================================================
    # NORMALIZED CONFUSION MATRIX
    # ========================================================

    normalized_cm = (
        overall_cm.astype(float) /
        overall_cm.sum(
            axis=1,
            keepdims=True
        )
    )

    print("\n")
    print(
        "=" * 80
    )

    print(
        "NORMALIZED CONFUSION MATRIX"
    )

    print(
        "=" * 80
    )

    np.set_printoptions(
        precision=4,
        suppress=True
    )

    print(
        normalized_cm
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = os.path.join(
        MODEL_OUTPUT_PATH,
        '10_fold_results.csv'
    )

    save_results_csv(
        fold_results,
        complexity,
        csv_path
    )

    # ========================================================
    # SAVE SUMMARY TXT
    # ========================================================

    summary_path = os.path.join(
        MODEL_OUTPUT_PATH,
        'final_results.txt'
    )

    with open(
        summary_path,
        'w'
    ) as f:

        f.write(
            "MALWARE IMAGE CLASSIFICATION RESULTS\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "10-FOLD CROSS VALIDATION\n\n"
        )

        f.write(
            f"Accuracy: "
            f"{np.mean(fold_accuracy):.4f} "
            f"+/- "
            f"{np.std(fold_accuracy):.4f}%\n"
        )

        f.write(
            f"Macro Precision: "
            f"{np.mean(fold_precision):.4f} "
            f"+/- "
            f"{np.std(fold_precision):.4f}%\n"
        )

        f.write(
            f"Macro Recall: "
            f"{np.mean(fold_recall):.4f} "
            f"+/- "
            f"{np.std(fold_recall):.4f}%\n"
        )

        f.write(
            f"Macro F1: "
            f"{np.mean(fold_f1_macro):.4f} "
            f"+/- "
            f"{np.std(fold_f1_macro):.4f}%\n"
        )

        f.write(
            f"Weighted F1: "
            f"{np.mean(fold_f1_weighted):.4f} "
            f"+/- "
            f"{np.std(fold_f1_weighted):.4f}%\n"
        )

        f.write(
            f"Latency: "
            f"{np.mean(fold_latency):.4f} "
            f"+/- "
            f"{np.std(fold_latency):.4f} ms\n"
        )

        f.write(
            f"Throughput: "
            f"{np.mean(fold_throughput):.4f} "
            f"+/- "
            f"{np.std(fold_throughput):.4f} images/sec\n"
        )

        f.write(
            "\nMODEL COMPLEXITY\n"
        )

        f.write(
            "=" * 80 + "\n"
        )

        if complexity is not None:

            f.write(
                f"Total Parameters: "
                f"{complexity['total_params']}\n"
            )

            f.write(
                f"Trainable Parameters: "
                f"{complexity['trainable_params']}\n"
            )

            f.write(
                f"Non-trainable Parameters: "
                f"{complexity['non_trainable_params']}\n"
            )

            f.write(
                f"Model Size: "
                f"{complexity['model_size_mb']:.4f} MB\n"
            )

            if complexity['flops'] is not None:

                f.write(
                    f"FLOPs: "
                    f"{complexity['flops']}\n"
                )

                f.write(
                    f"GFLOPs: "
                    f"{complexity['flops'] / 1e9:.6f}\n"
                )

                f.write(
                    f"MACs: "
                    f"{complexity['macs']}\n"
                )

                f.write(
                    f"MMACs: "
                    f"{complexity['macs'] / 1e6:.6f}\n"
                )

        f.write(
            "\nAGGREGATED CONFUSION MATRIX\n"
        )

        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            str(overall_cm)
        )

    print(
        f"\nFinal summary saved to:\n"
        f"{summary_path}"
    )

    print("\n")
    print(
        "#" * 100
    )

    print(
        "EXPERIMENT COMPLETED SUCCESSFULLY"
    )

    print(
        "#" * 100
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    train()
