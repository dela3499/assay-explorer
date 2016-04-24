import os

from utils import maybe_get_model, persist_widget_value


PATH = os.getcwd()
DB_PATH = os.path.join(PATH, 'db', 'db.csv')
THRESHOLD_FILEPATH = os.path.join(PATH, 'tmp', 'temp_thresholds.json')
ALL_THRESHOLDS_FILEPATH = os.path.join(PATH, 'tmp', 'cell_phase_thresholds.csv')
MODEL_FILEPATH = os.path.join(PATH, 'tmp', 'threshold-ui-model.json')

uiget = maybe_get_model(MODEL_FILEPATH)  # key -> value
uiset = persist_widget_value(MODEL_FILEPATH)      # widget -> key

