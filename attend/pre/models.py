

def resnet(layer=None):
    import tensorflow.contrib.keras as K
    model = K.applications.ResNet50(
            include_top=False,
            weights='imagenet')

    if layer:
        out_layer = model.get_layer(layer).output
        model = K.models.Model(model.input, out_layer)

    return model


def vggface(pooling=None):
    import tensorflow.contrib.keras as K
    from keras_vggface.vggface import VGGFace

    model = VGGFace(include_top=False,
                    pooling=pooling,
                    input_shape=(224,224,3))

    return model

AVAILABLE_MODELS = ['vggface_conv', 'vggface_avg', 'vggface_max',
                    'resnet_conv', 'resnet_avg', 'resnet_max',
                    'resnet_act']

def model_for(key):
    assert key in AVAILABLE_MODELS, 'Unknown model key'

    model, kind = key.split('_')

    if model == 'vggface':
        if kind == 'conv' or kind == 'none':
            kind = None
        model = vggface(pooling=kind)
        return model
    else:
        raise Exception('Not implemented yet')


