import inspect


def pick(dic, l):
    '''
    Pick from dictionary all the keys present in a list of keywords
    After the spirit of Lodash pick
    '''
    return { k: v for k, v in dic.items() if k in l }


def params_for(fun):
    return list(inspect.signature(fun).parameters)
