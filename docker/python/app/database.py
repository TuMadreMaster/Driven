#/usr/bin/python3

if __name__ == "__main__":
    import logic
else:
    from . import logic

from sqlalchemy import Table, MetaData, create_engine, insert, select

import pandas as pd
import numpy as np

import logging

log = logging.getLogger(__name__)

class Database:

    """PostgreSQL Database class."""

    def __init__(
            self,
            user = "lis",
            pswd = "Lis2568djcxdnsan45n4nrnenf",
            name = "seinsa",
            host = "100.10.10.4",
            port = "5432",   
            ) -> None:

        self.user = user
        self.pswd = pswd
        self.name = name
        self.host = host
        self.port = port
        print('Connecting to database...')

        self.engine = create_engine(f"postgresql://{self.user}:{self.pswd}@{self.host}:{self.port}/{self.name}")
        try:
            with self.engine.connect():
                log.info("Connection test successful!")
        except Exception as exception:
            log.exception("Can't connect to the database!")
            raise exception
        
        self.meta = MetaData()

        self.data_table = Table("automatica_largoiko", self.meta, autoload=True, autoload_with=self.engine)
        self.meta_table = Table("metadata", self.meta, autoload=True, autoload_with=self.engine)
        self.pred_table = Table("predictions", self.meta, autoload=True, autoload_with=self.engine)
        self.train_table = Table("train_data", self.meta, autoload=True, autoload_with=self.engine)
        self.fcast_table = Table("fcast_data", self.meta, autoload=True, autoload_with=self.engine)

    def select_train(self) -> pd.DataFrame:

        query = select(self.data_table)
        with self.engine.begin() as conn:
            df: pd.DataFrame = pd.read_sql(str(query.compile(self.engine)), con=conn)
        # convert all nan and empty strings into None
        # (None is also present in itself)        
        df = df.astype(object).replace(np.nan, 'None')
        df = df.astype(object).replace('', 'None')
        return df

    def insert_metadata(self, model: logic.Transformer) -> None:

        meta = model.meta
        query = insert(self.meta_table).values(
            model=str(model.folder.name), 
            timestamp=meta["timestamp"],
            sample_size=meta["sample_size"],
            mask_mode=meta["mask_mode"],
            mask_lda=meta["mask_lda"],
            detect_thresh=meta["detection_threshold"],
            class_thresh=meta["classification_threshold"],
            test_acc=meta["test_accuracy"],
            test_prec=meta["test_precision"],
            test_rec=meta["test_recall"],
            test_f1=meta["test_f1"],
            test_brier=meta["test_brier"],
            test_roc=meta["test_roc"],
            test_tpr=meta["test_true_positives"],
            test_fpr=meta["test_false_positives"],
            test_tnr=meta["test_true_negatives"],
            test_fnr=meta["test_false_negatives"],
            nsamp_total=meta["nsamples_total"],
            nsamp_positive=meta["nsamples_positive"],
            nsamp_negative=meta["nsamples_negative"],
            nsamp_train=meta["nsamples_train"],
            nsamp_eval=meta["nsamples_eval"],
            nsamp_test=meta["nsamples_test"],
            nepochs=meta["n_epochs"]) 
        with self.engine.begin() as conn:
            conn.execute(query)
        pass

    def insert_train_data(self, df: pd.DataFrame) -> int:
        with self.engine.begin() as conn:
            rows = df.to_sql(
                name=self.train_table.name,
                con = conn, index=False, 
                if_exists="append", 
                method="multi")
        return rows

    def insert_fcast_data(self, df: pd.DataFrame) -> int:
        with self.engine.begin() as conn:
            rows = df.to_sql(
                name=self.fcast_table.name,
                con = conn, index=False, 
                if_exists="append", 
                method="multi")
        return rows

    def select_predict(self, sample_size: int) -> pd.DataFrame:

        query = select(self.data_table).order_by(self.data_table.c.time.desc()).limit(sample_size)
        with self.engine.begin() as conn:
            data: pd.DataFrame = pd.read_sql(str(query.compile(self.engine, 
                compile_kwargs={"literal_binds": True})), con=conn)
        data.sort_values("time", ascending=True, inplace=True)
        return data


    def insert_predict(self, time, rof, model) -> None:

        query = insert(self.pred_table).values(ts=str(time[0]), rof=float(rof[0][0]), model=model) 
        with self.engine.begin() as conn:
            conn.execute(query)

if __name__ == "__main__":
    db = Database(host="0.0.0.0", port="5030")
    logging.basicConfig(
        # filename=logfile,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    pass

    
