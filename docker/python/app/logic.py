#/usr/bin/python3

if __name__ == "__main__":
    import network
    import cleanup
    import mining
else:
    from . import network
    from . import cleanup
    from . import mining

# sklearn imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

# keras imports
from keras.callbacks import EarlyStopping
from keras.models import load_model
import keras_tuner as kt
from keras import Model

# general imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml

# stdlib imports
from datetime import datetime
from pathlib import Path
import logging


# ================================================================= #

class Transformer():

    RANDOM_STATE: int = 42

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def __init__(self, folder: Path) -> None:

        self.folder = folder
        self.model_file: Path = folder / "model.h5"
        self.meta_file: Path = folder / "meta.yml"
        self.log_file: Path = folder / "log.txt"
        self.hp_file: Path = folder / "hp.yml"

        # set up logging
        self.log = logging.getLogger(folder.name)
        self.log.setLevel(logging.INFO)

        log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        log_fh = logging.FileHandler(self.log_file)
        log_fh.setLevel(logging.DEBUG)
        log_fh.setFormatter(log_fmt)
        self.log.addHandler(log_fh)

        # load training meta if available
        if self.meta_file.exists():
            with open(self.meta_file, "r") as file:
                self.meta: dict = yaml.load(file, Loader=yaml.FullLoader)
        else:
            self.meta = dict()

        # load model if saved
        self.model: Model = load_model(self.model_file) if self.model_file.exists() else None
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def train(self, 
            # data generation
            data_dump: pd.DataFrame,
            sample_size: int = 20,
            mask_mode: str = 'arbitrary',
            mask_lda: float = 60.0,
            skip_rows: int = 35,
            skip_sessions: int = 1,
            # training
            max_epochs: int = 50,
            batch_size: int = 1,
            shuffle: bool = True, 
            test_split: float = 0.1,
            eval_split: float = 0.2,
            # analysis
            class_thresh: float = 0.5,
            plots: bool = False,
            plots_dpi: int = 250,
            max_shifts: int = 200,
            ) -> None: 

        self.meta["timestamp"] = datetime.now()

        # register settings as metadata
        self.meta["sample_size"] = sample_size
        self.meta["mask_mode"] = mask_mode
        self.meta["mask_lda"] = mask_lda
        self.meta["skip_rows"] = skip_rows
        self.meta["skip_sessions"] = skip_sessions
        self.meta["max_epochs"] = max_epochs
        self.meta["batch_size"] = batch_size
        self.meta["shuffle"] = shuffle
        self.meta["test_split"] = test_split
        self.meta["eval_split"] = eval_split
        self.meta["classification_threshold"] = class_thresh
        self.meta["max_shifts"] = max_shifts

        # process the data dump
        self.log.info("CLEANUP PHASE")
        X, y, vars = self._data(
            df=data_dump,
            sample_size=sample_size,
            mask_mode=mask_mode,
            mask_lda=mask_lda,
            skip_rows=skip_rows,
            skip_sessions=skip_sessions
        )
        X: np.ndarray
        y: np.ndarray
        vars: list[str]
        input_shape = list(X.shape[1:])

        self.log.info("TRAIN PHASE")
        # shuffle and split into train evaluation and test sets
        self.log.info("Partitioning train/eval/test sets...")
        X_traineval, X_test, y_traineval, y_test = train_test_split(
            X, y, test_size=test_split, shuffle=shuffle, random_state=self.RANDOM_STATE)
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_traineval, y_traineval, test_size=eval_split, shuffle=shuffle, random_state=self.RANDOM_STATE)

        # add data info to meta
        self.meta["vars"] = vars
        self.meta["nvars"] = len(vars)
        self.meta["nsamples_total"] = len(y)
        self.meta["nsamples_train"] = len(y_train)
        self.meta["nsamples_eval"] = len(y_eval)
        self.meta["nsamples_test"] = len(y_test)

        # train the transformer
        self.log.info("Starting training procedure... (see log_train.csv)")
        self.model = network.transformer_model(input_shape=input_shape)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_eval, y_eval),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True)
            ],
        )

        # save the weights
        self.log.info(f"Saving model to '{self.model_file}'...")
        self.model.save(self.model_file)

        n_epochs = len(history.history['loss'])
        self.meta["n_epochs"] = n_epochs
        df_train = pd.DataFrame.from_dict({
            "model": [str(self.folder.name)]*n_epochs,
            "epoch": np.arange(n_epochs),
            "loss_train": history.history['loss'],
            "loss_eval": history.history['val_loss'],
            "roc_train": history.history['roc'],
            "roc_eval": history.history['val_roc'],
            "pr_train": history.history['pr'],
            "pr_eval": history.history['val_pr']})

        # save meta dict
        self.log.info(f"Updating metadata file...")
        with open(self.meta_file, "w") as file:
            yaml.dump(self.meta, file, indent=2)

        if plots:

            self.log.info("Generating training plots...")
            
            plt.figure("loss", constrained_layout=True) # loss
            plt.plot(history.history['loss'], label="train")
            plt.plot(history.history['val_loss'], label="eval")
            plt.title("Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.grid(True)
            plt.legend()
            plt.savefig(self.folder / "train_eval_loss.png", dpi=plots_dpi)
            plt.close()

            plt.figure("roc", constrained_layout=True)  # roc
            plt.plot(history.history['roc'], label="train")
            plt.plot(history.history['val_roc'], label="eval")
            plt.title("Receiver Operating Characteristic (ROC) AUC")
            plt.ylabel("ROC AUC")
            plt.xlabel("Epoch")
            plt.grid(True)
            plt.legend()
            plt.savefig(self.folder / "train_eval_roc.png", dpi=plots_dpi)
            plt.close()

            plt.figure("loss", constrained_layout=True) # pr
            plt.plot(history.history['pr'], label="train")
            plt.plot(history.history['val_pr'], label="eval")
            plt.title("Precision-Recall AUC")
            plt.ylabel("PR AUC")
            plt.xlabel("Epoch")
            plt.grid(True)
            plt.legend()
            plt.savefig(self.folder / "train_eval_pr.png", dpi=plots_dpi)
            plt.close()
        
        # test model performance
        self.log.info("TEST PHASE")
        self._test(
            X_test=X_test,
            y_test=y_test,
            class_thresh=class_thresh,
            plots=plots,
            plots_dpi=plots_dpi
        )

        self.log.info("FORECAST PHASE")
        df_fcast = self._fcast(
            df=data_dump,
            sample_size=sample_size,
            mask_mode=mask_mode,
            mask_lda=mask_lda,
            skip_rows=skip_rows,
            skip_sessions=skip_sessions,
            max_shifts=max_shifts,
            class_thresh=class_thresh,
        )

        # save meta dict
        self.log.info(f"Saving model metadata...")
        with open(self.meta_file, "w") as file:
            yaml.dump(self.meta, file, indent=2)

        return df_train, df_fcast

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def _data(self,
            df: pd.DataFrame,
            sample_size: int = 20,
            mask_mode: str = 'arbitrary',
            mask_lda: float = 60.0,
            skip_rows: int = 35,
            skip_sessions: int = 1,
            pickle: Path = None
            ) -> np.ndarray:

        """ Generates datasets from the training dump. """

        # save settings in the meta dict
        
        # CLEANUP
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        self.log.info("Cleaning up data dump...")

        # add time deltas
        df = cleanup.add_time_deltas(df)

        # remove first rows (dataset-specific)
        df = cleanup.remove_first_rows(df, skip_rows)

        # remove first product chunk (defective)
        df = cleanup.remove_first_product_chunks(df, skip_sessions)

        # assign correct datatypes and categorize columns
        df = cleanup.assing_correct_dtypes(df)

        # drop suspicious columns
        df = cleanup.drop_antropic_cols(df)

        # TRAINING SET GENERATION
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
        self.log.info("Generating training sets...")

        # event threshold
        mask, threshold = mining.generate_anomaly_mask(df, lda=mask_lda, mode=mask_mode)
        self.meta["detection_threshold"] = threshold

        # generate the events
        target_events, target_ts, target_mask = mining.generate_target_sets_by_events(df, mask, sample_size)
        self.meta["nsamples_positive"] = len(target_ts)
        blank_ts, blank_mask = mining.generate_blank_sets_by_events(df, mask, sample_size)
        self.meta["nsamples_negative"] = len(blank_ts)

        # check columns used for predictions
        columns = cleanup.convert_dataset_to_array(target_ts[0], return_cols=True)
        
        # generate x training variable
        target_array_ts = np.array([cleanup.convert_dataset_to_array(ts) for ts in target_ts])
        blank_array_ts = np.array([cleanup.convert_dataset_to_array(ts) for ts in blank_ts])
        X: np.ndarray = np.concatenate((target_array_ts, blank_array_ts), axis=0)

        # generate y training variable
        y_target = np.ones(len(target_ts), dtype=np.int8)
        y_blank = np.zeros(len(blank_ts), dtype=np.int8)
        y: np.ndarray = np.concatenate((y_target, y_blank))
        
        # save training data
        return X, y, columns

    def _test(self,
            X_test: np.ndarray,
            y_test: np.ndarray,
            class_thresh: float,
            plots: bool,
            plots_dpi: int
            ) -> None:

        # generate test predictions
        self.log.info("Generating test set predictions...")

        y_prob = self.model.predict(X_test)  # probabilities
        y_pred = np.zeros_like(y_prob)  # binary predict
        y_pred[y_prob > class_thresh] = 1

        self.log.info("Calculating test metrics...")

        CM = confusion_matrix(y_test, y_pred, normalize="true")

        self.meta["test_true_negatives"] = float(CM[0][0])
        self.meta["test_false_negatives"] = float(CM[1][0])
        self.meta["test_true_positives"] = float(CM[1][1])
        self.meta["test_false_positives"] = float(CM[0][1])

        self.meta["test_accuracy"] = float(accuracy_score(y_test, y_pred))
        self.meta["test_precision"] = float(precision_score(y_test, y_pred))
        self.meta["test_recall"] = float(recall_score(y_test, y_pred))
        self.meta["test_f1"] = float(f1_score(y_test, y_pred))
        self.meta["test_brier"] =  float(brier_score_loss(y_test, y_prob))
        self.meta["test_roc"] = float(roc_auc_score(y_test, y_prob))

        roc_fpr, roc_tpr, _ = roc_curve(y_test,y_prob)

        if plots:  
            plt.figure("ROC", constrained_layout=True)
            plt.plot(roc_fpr, roc_tpr, label="ROC Curve")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.grid(True)
            plt.savefig(self.folder / "test_roc_curve.png", dpi=plots_dpi)
            plt.close()

        # precision-recall metric
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
        
        if plots:
            plt.figure("PR", constrained_layout=True)
            plt.title("Precision-Recall (PR) Curve")
            plt.plot(pr_rec, pr_prec, label="PR Curve")
            plt.ylabel("Precision")
            plt.xlabel("Recall")
            plt.grid(True)
            plt.savefig(self.folder / "test_pr_curve.png", dpi=plots_dpi)
            plt.close()

    def _fcast(self, 
            df: pd.DataFrame,
            sample_size: int,
            mask_mode: str,
            mask_lda: float,
            skip_rows: int,
            skip_sessions: int,
            max_shifts: int,
            class_thresh: float,
            ) -> pd.DataFrame:

        df = cleanup.add_time_deltas(df)

        # remove first rows (dataset-specific)
        df = cleanup.remove_first_rows(df, skip_rows)

        # remove first product chunk (defective)
        df = cleanup.remove_first_product_chunks(df, skip_sessions)

        # assign correct datatypes and categorize columns
        df = cleanup.assing_correct_dtypes(df)

        # drop suspicious columns
        df = cleanup.drop_antropic_cols(df)

        mask, _ = mining.generate_anomaly_mask(df, lda=mask_lda, mode=mask_mode)

        model = self.folder.name
        df_fcast = pd.DataFrame(
            columns=["model", "shift","prob_avg","prob_std","acc","brier"])

        for lag in range(max_shifts):

            _, target_ts, _ = mining.generate_target_sets_by_events(df, mask, sample_size, sample_shift=lag)
            target_array_ts = np.array([cleanup.convert_dataset_to_array(ts) for ts in target_ts])

            y_prob = self.model.predict(target_array_ts)  # probabilities
            y_pred = np.zeros_like(y_prob)  # binary predict
            y_pred[y_prob > class_thresh] = 1

            # correct prediction
            y_true = np.zeros_like(y_pred) if lag else np.ones_like(y_pred)

            prob_avg = np.mean(y_prob)
            prob_std = np.std(y_prob)

            acc = accuracy_score(y_true, y_pred)
            brier = brier_score_loss(y_true, y_prob)

            # log data
            df_fcast.loc[lag] = [model, lag, prob_avg, prob_std, acc, brier]

        # TODO create som plots?

        return df_fcast

    # TODO fix this method
    def tune_model(self,
            # data generation
            data_dump: pd.DataFrame,
            sample_size: int = 20,
            mask_mode: str = 'arbitrary',
            mask_lda: float = 60.0,
            skip_rows: int = 35,
            skip_sessions: int = 1,
            shuffle: bool = True,
            pickle: Path = None
            ) -> None: 

        # process the data dump
        self.log.info("CLEANUP PHASE")
        X, y, vars = self._data(
            df=data_dump,
            sample_size=sample_size,
            mask_mode=mask_mode,
            mask_lda=mask_lda,
            skip_rows=skip_rows,
            skip_sessions=skip_sessions,
            pickle=pickle
        )
        X: np.ndarray
        y: np.ndarray
        vars: list[str]
        input_shape = list(X.shape[1:])

        # shuffle and split into 
        self.log.info("Partitioning train/eval/test sets...")
        X_traineval, X_test, y_traineval, y_test = train_test_split(
            X, y, test_size=0.1, shuffle=shuffle, random_state=self.RANDOM_STATE)
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_traineval, y_traineval, test_size=0.2, shuffle=shuffle, random_state=self.RANDOM_STATE)

        # training metadata
        self.meta["shuffle"] = shuffle
        self.meta["nsamples_train"] = len(y_train)
        self.meta["nsamples_eval"] = len(y_eval)
        self.meta["nsamples_test"] = len(y_test)

        # define the hypermodel tuner
        self.log.info("Build the model tuner...")
        model_builder = network.transformer_builder_input(input_shape)
        tuner = kt.Hyperband(model_builder,
            objective=kt.Objective("pr", direction="max"),
            max_epochs=20,
            factor=3,
            directory='data/tuning', # fix relative path
            project_name='transformer'
            )
        
        # train the hypermodel
        self.log.info("Starting the tuning process...")
        tuner.search(
            X_train, y_train,
            validation_data=(X_eval, y_eval),
            epochs=200,
            batch_size=1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        )

        # save the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        with open(self.hp_file, "w") as file:
            yaml.dump(best_hps.values, file, indent=2)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def predict(self, df: pd.DataFrame) -> float:

        # assign correct datatypes and categorize columns
        df = cleanup.assing_correct_dtypes(df)

        # drop suspicious columns
        df = cleanup.drop_antropic_cols(df)

        # split into different inputs
        samples, times = cleanup.split_into_samples(df, self.meta["sample_size"])

        # convert to datapoints
        data = np.array([cleanup.convert_dataset_to_array(s) for s in samples])
        rof = self.model.predict(data) 

        return rof, times

if __name__ == "__main__":

    test_fold = Path(__file__).parent.parent.resolve() / "models" / "MO_2000-test"
    test_fold.mkdir(parents=True, exist_ok=True)

    import os 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    import pickle
    with open(Path(__file__).parent.resolve() / "data7.pkl", "rb") as f:
        data: pd.DataFrame = pickle.load(f)

    model = Transformer(test_fold)

    log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_ch = logging.StreamHandler()
    log_ch.setLevel(logging.DEBUG)
    log_ch.setFormatter(log_fmt)
    model.log.addHandler(log_ch)

    df_meta, df_train, df_fcast = model.train(data_dump=data, max_epochs=1)

    if True:
        pass

    pass