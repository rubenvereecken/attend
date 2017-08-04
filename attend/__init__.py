

class GraphKeys:
    INPUT_RUNNERS = 'input_runners'
    VAL_INPUT_RUNNERS = 'val_input_runners'
    VAL_SUMMARIES = 'val_summaries'
    STATS_SUMMARIES = 'stats_summaries'
    INITIAL_STATES = 'initial_states'


from .report import SummaryProducer
from .run import Runner
from .log import Log
from .solver import AttendSolver
from .model import AttendModel
from .encoder import Encoder
from .evaluate import Evaluator


TIMESTAMP_FORMAT = "%d-%m-%Y-%H-%M-%S"
