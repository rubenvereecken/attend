import logging
import sys
import os
import cson

from attend import util

dir_path = os.path.dirname(os.path.realpath(__file__))

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
    def _get_logger(cls, name):
        if cls.root_logger is None:
            cls.root_logger = logging.getLogger()
            cls.root_logger.warning('No root logger set! Falling back')

        return cls.root_logger.getChild(name)

    @classmethod
    def get_logger(cls, name):
        import lazy_object_proxy as lazy
        from functools import partial
        return lazy.Proxy(partial(Log._get_logger, name))


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
        git_dir = dir_path + '/../.git'
        git_logs_file = git_dir + '/logs/HEAD'
        try:
            sha = subprocess.check_output(['git', '--git-dir', git_dir,
                                           '--no-pager', 'log', '--oneline', '-n1'])
            file_path = cls.log_dir + '/{}'.format(file_name)
            with open(file_path, 'w') as f:
                f.write(sha.decode())
        except Exception as e:
            print('Failed to get git version, skipping')

    @classmethod
    def save_hostname(cls, file_name='host'):
        import socket
        hostname = socket.gethostname()
        file_path = cls.log_dir + '/{}'.format(file_name)
        with open(file_path, 'w') as f:
            f.write(str(hostname))

    @classmethod
    def save_meta(cls, d, file_name='meta'):
        file_path = cls.log_dir + '/{}.cson'.format(file_name)
        with open(file_path, 'w') as f:
            cson.dump(dict(d), f, sort_keys=True, indent=4)

    @classmethod
    def save_condor(cls, file_name='classad.condor'):
        import subprocess
        file_path = cls.log_dir + '/{}'.format(file_name)
        try:
            ad = subprocess.check_output(['condor_status', '$(hostname)', '-l'])
            ad = ad.decode()
            with open(file_path + '.unsorted', 'w') as f:
                f.write(ad)

            ad_lines = list(filter(lambda x: len(x) > 0, ad.split('\n')))
            ad = '\n'.join(sorted(ad_lines))
            with open(file_path, 'w') as f:
                f.write(ad)
        except:
            print('Not a condor machine or something went wrong')

    @classmethod
    def last_logdir(cls, path, prefix=None):
        import re
        r = re.compile('^((?P<prefix>.*?)_)?(?P<time_stamp>[0-9]{2}-[0-9]{2}-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2})')
        dirs = os.listdir(path)
        matches = zip(map(lambda s: r.match(s), dirs), dirs)
        matches = filter(lambda x: x[0], matches)
        matches = map(lambda x: (x[0].groupdict(), x[1]), matches)

        if not prefix is None:
            matches = filter(lambda m: m[0]['prefix'] == prefix, matches)

        matches = map(lambda s: (util.parse_timestamp(s[0]['time_stamp']), s[1]), matches)
        matches = sorted(matches, key=lambda x: x[0])

        if len(matches) > 0:
            return path + '/' + matches[-1][1]

        return None

    @classmethod
    def get_args(cls, path, filename='args.cson'):
        with open(path + '/' + filename, 'r') as f:
            args = cson.load(f)
        return args


