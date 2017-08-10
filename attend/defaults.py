
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
        encode_hidden_units       = 256,
        # dense_layer             = 1,
        dense_spec                = '-,relu',

        update_rule               = 'adam',

        use_dropout               = True,
        dropout                   = .75,

        shuffle_examples          = True,
        shuffle_examples_capacity = None,

        shuffle_splits            = False,
        shuffle_splits_capacity   = None,

        stats_every               = 100,
        gen_log_dir               = True,
        save_eval_graph           = True,

        attention_units           = 512,
        final_sigmoid             = False,
        learn_initial_states      = True
    )

    debug_params = dict(
        shuffle_examples = True,
        encode_lstm      = False,
        save_eval_graph  = False
        )

    def __init__(self, debug=False):
        self.__dict__ = Defaults.common_params.copy()
        if debug:
            self.__dict__.update(Defaults.debug_params)
