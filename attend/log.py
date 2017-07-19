import logging
import sys

class Log:
    filename = None
    level = None

    @classmethod
    def setup(cls, filename, debug=True):
        cls.filename = filename
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
        return cls.root_logger.getChild(name)
