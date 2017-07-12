import inspect


def pick(dic, l):
    '''
    Pick from dictionary all the keys present in a list of keywords
    After the spirit of Lodash pick
    '''
    return { k: v for k, v in dic.items() if k in l }


def params_for(fun):
    return list(inspect.signature(fun).parameters)


def call_with(fun, d):
    return fun(**pick(d, params_for(fun)))


def init_with(cls, d):
    return cls(**pick(d, params_for(cls.__init__)))


def notify(title, text='', duration=5000):
    import subprocess as s
    s.call(['notify-send', title, text])
