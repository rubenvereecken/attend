
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
        dense_spec                = '-,relu',

        update_rule               = 'adam',

        use_dropout               = False,
        dropout                   = .75,
        use_maxnorm = False,
        use_batch_norm = True,
        # NOTE if training good but validation low, lower decay
        batch_norm_decay = 0.9, # Default 0.99

        shuffle_examples          = True,
        # Defaults to batch_size * 4
        shuffle_examples_capacity = None,

        stats_every               = 100,
        gen_log_dir               = True,
        save_eval_graph           = True,

        attention_units           = 512,
        learn_initial_states      = True,
        final_activation = 'tanh',

        # Scheduled sampling
        sampling_scheme = 'inverse_sigmoid',
        sampling_min = .75
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
