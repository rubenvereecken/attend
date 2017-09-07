import demandimport
demandimport.ignore('tensorflow')
# demandimport.enable()


class GraphKeys:
    INPUT_RUNNERS     = 'input_runners'
    VAL_INPUT_RUNNERS = 'val_input_runners'
    VAL_SUMMARIES     = 'val_summaries'
    STATS_SUMMARIES   = 'stats_summaries'
    INITIAL_STATES    = 'initial_states'
    INPUT             = 'input'
    # STATE             = 'state'
    STATE_SAVER       = 'state_saver'
    STATE_RESET       = 'state_reset' # Holds state_saver.reset_states()
    OUTPUT = 'output'
    INTERNAL = 'internal'
    ERRORS = 'errors'
    CONTEXT = 'context'
    SAMPLING_EPSILON = 'epsilon'


# import pdb; pdb.set_trace()  # XXX BREAKPOINT

from .report import SummaryProducer
from .run import Runner
from .log import Log
from .solver import AttendSolver
from .encoder import Encoder
from .common import *


TIMESTAMP_FORMAT = "%d-%m-%Y-%H-%M-%S"
