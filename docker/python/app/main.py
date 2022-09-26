#/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from .logic import Transformer
from .database import Database

from fastapi import BackgroundTasks
from fastapi import HTTPException, status
from fastapi import FastAPI
import uvicorn

from datetime import datetime
from pathlib import Path
from enum import Enum
import logging

# SETTINGS
# ================================================================= #

log = logging.getLogger(__name__)
logging.basicConfig(
    # filename=logfile,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# connect to the database
DB = Database()

# base folder where all models are stored
MODELS_FOLDER = Path(__file__).parent.resolve() / "models" 
MODEL_PATHS = sorted([f.resolve() for f in MODELS_FOLDER.iterdir() if f.is_dir()])

# load latest model
global MODEL
MODEL = Transformer(MODEL_PATHS[-1]) if MODEL_PATHS else None
TRAIN_MODEL = None

app = FastAPI(
    title="DIGITBrain Predictive Model",
    version="1.0",
    contact={
        "name": "LIS Data Solutions",
        "url": "https://www.lisdatasolutions.com",
        "email": "info@lisdatasolutions.com",
    }
)

@app.get('/')
def root():

    """
    Placeholder to check if connection is successful.
    """

    return "Conexión realizada con exito!"

@app.get('/status')
def model():

    """
    Returns the operational status of the service.
    """
    global MODEL
    global TRAIN_MODEL

    active_model = None if MODEL is None else str(MODEL.folder.name)
    training_model = None if TRAIN_MODEL is None else str(TRAIN_MODEL.folder.name)
        
    return {"active_model": active_model, "under_training": training_model}

class MaskMode(str, Enum):
    arbitrary = "arbitrary"
    median = "median"
    mean = "mean"

@app.get('/train', status_code=status.HTTP_202_ACCEPTED)
def train(
        bg_tasks: BackgroundTasks,
        # data generation
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
    global TRAIN_MODEL
    global MODEL

    """
    Grabs all the data in Timescale DB and trains a predictive model.
    """

    # check if model is being trained
    if TRAIN_MODEL is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Already training a model!")

    # create new model folder
    date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    mod_folder = MODELS_FOLDER / f"MO_{date}"
    if mod_folder.exists(): # check if it already exists
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Model already exists!")
    mod_folder.mkdir(parents=True, exist_ok=False)

    # training task
    def model_training():
        global TRAIN_MODEL
        global MODEL

        TRAIN_MODEL  = Transformer(mod_folder)      # create new model 
        df = DB.select_train()                      # grab datadump from db
        df_train, df_fcast = TRAIN_MODEL.train(     # create training data
            data_dump=df,
            sample_size=sample_size,
            mask_mode=str(mask_mode),
            mask_lda=mask_lda,
            skip_rows=skip_rows,
            skip_sessions=skip_sessions,
            shuffle=shuffle,
            max_epochs=max_epochs,
            batch_size=batch_size,
            test_split=test_split,
            eval_split=eval_split,
            class_thresh=class_thresh,
            plots=plots,
            plots_dpi=plots_dpi,
            max_shifts=max_shifts
        )
        # DB.insert_metadata(TRAIN_MODEL)       # save test data
        DB.insert_train_data(df_train)          # save train data
        DB.insert_fcast_data(df_fcast)          # save fcast data  
        MODEL = TRAIN_MODEL                     # use new model
        TRAIN_MODEL = None
        
        # TODO notify in some way?

    bg_tasks.add_task(model_training)
    
    return {"model_name": str(mod_folder.name), "path": str(mod_folder)}

@app.get('/predict', status_code=status.HTTP_201_CREATED)
def predict() -> None:

    """
    Grabs the latest data sample from Timescale DB, 
    makes a prediction and uploads it to the database.
    """

    # read data from pickle directly
    df = DB.select_predict(sample_size=MODEL.meta['sample_size'])
    rof, time = MODEL.predict(df)
    DB.insert_predict(rof=rof, time=time, model=str(MODEL.folder.name))
    return {"ts": str(time[0]), "rof": float(rof[0][0])}

@app.get('/logs')
def logs() -> None:

    """
    Returns the train log of the active/in-training model.
    """

    target = MODEL if TRAIN_MODEL is None else TRAIN_MODEL
    if target is None:
        return None
    if not target.log_file.exists():
        return "Log file not available."
    with open(target.log_file, "r") as f:
        return f.read().split('\n')

@app.get('/meta')
def logs() -> None:

    """
    Returns metadata related to the current model.
    """

    # return training logs
    if MODEL is None:
        return None
    return MODEL.meta

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)