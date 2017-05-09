from collections import Iterable

from fuel.schemes import IterationScheme
from picklable_itertools import cycle, imap, xrange
from picklable_itertools.extras import partition_all

class InfiniteSequentialBatchIterator(IterationScheme):
    '''
    When I say infinite I mean infinite.
    '''

    requests_examples = False # Batched

    def __init__(self, examples, batch_size):
        # This bit makes sure indices is always a list of sorts
        if isinstance(examples, Iterable):
            self.indices = examples
        else:
            self.indices = xrange(examples)

        self.batch_size = batch_size


    def get_request_iterator(self):
        '''
        Careful this is indeed infinite
        '''
        return imap(list, partition_all(self.batch_size, cycle(self.indices)))
