import logging
import sys
import cson

class Log:
    filename = None
    log_dir = None
    level = None
    root_logger = None

    @classmethod
    def setup(cls, log_dir, debug=True):
        cls.log_dir = log_dir
        filename = log_dir + '/log.txt'
        cls.filename = filename
        cls.debug = debug
        cls.level = logging.DEBUG if debug else logging.INFO
        file_formatter = logging.Formatter('%(asctime)s - %(name)-20s - %(levelname)-8s : %(message)s')
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(file_formatter)

        stream_formatter = logging.Formatter('%(asctime)s [%(name)-20s] [%(levelname)-8s] : %(message)s',
                datefmt='%H:%M:%S')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)

        cls.root_logger = logging.getLogger()
        cls.root_logger.setLevel(cls.level)
        cls.root_logger.addHandler(file_handler)
        cls.root_logger.addHandler(stream_handler)


    @classmethod
    def get_logger(cls, name):
        if cls.root_logger is None:
            cls.root_logger = logging.getLogger()
            cls.root_logger.warning('No root logger set! Falling back')

        return cls.root_logger.getChild(name)


    @classmethod
    def save_args(cls, args, file_name='args'):
        if not isinstance(args, dict):
            args = args.__dict__
        file_path = cls.log_dir + '/{}.cson'.format(file_name)
        with open(file_path, 'w') as f:
            cson.dump(args, f, sort_keys=True, indent=4)

    @classmethod
    def save_pid(cls, file_name='pid'):
        import os
        file_path = cls.log_dir + '/{}'.format(file_name)
        with open(file_path, 'w') as f:
            f.write(str(os.getpid()))

    @classmethod
    def save_env(cls, file_name='env'):
        import os
        file_path = cls.log_dir + '/{}.cson'.format(file_name)
        with open(file_path, 'w') as f:
            cson.dump(dict(os.environ.items()), f, sort_keys=True, indent=4)

    @classmethod
    def save_git_version(cls, file_name='gitversion'):
        import subprocess
        sha = subprocess.check_output(['git', 'describe', '--always'])
        file_path = cls.log_dir + '/{}'.format(file_name)
        with open(file_path, 'w') as f:
            f.write(str(sha))

    @classmethod
    def save_hostname(cls, file_name='host'):
        import socket
        hostname = socket.gethostname()
        file_path = cls.log_dir + '/{}'.format(hostname)
        with open(file_path, 'w') as f:
            f.write(str(hostname))

    @classmethod
    def save_meta(cls, d, file_name='meta'):
        file_path = cls.log_dir + '/{}.cson'.format(file_name)
        with open(file_path, 'w') as f:
            cson.dump(dict(d), f, sort_keys=True, indent=4)
