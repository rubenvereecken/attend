
class Defaults:
    common_params = dict(
        batch_size                = 12,
        val_batch_size            = None,
        steps_per_epoch           = 1000,
        num_epochs                = 50,
        learning_rate             = 0.001,
        time_steps                = 100,
        num_hidden                = 512,
        conv_impl                 = 'none',
        encode_lstm               = True,
        encode_hidden_units       = 512,
        dense_layer               = 1,

        update_rule               = 'adam',

        shuffle_examples          = True,
        shuffle_examples_capacity = None,

        shuffle_splits            = False,
        shuffle_splits_capacity   = None,

        stats_every = 100,
        gen_log_dir = True
    )

    debug_params = dict(
        shuffle_examples = True,
        encode_lstm = False
        )

    def __init__(self, debug=False):
        self.__dict__ = Defaults.common_params.copy()
        if debug:
            self.__dict__.update(Defaults.debug_params)
