

class CombineCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()  # sin kwargs

    def on_epoch_end(self, epoch, logs=None):
        import numpy as np
        logs = logs or {}
        acc = logs.get('val_Accuracy') or logs.get('val_accuracy')
        loss = logs.get('val_loss')

        if (acc is not None) and (loss is not None) and np.isfinite(loss) and (loss != 0):
            logs['combine_metric'] = float(acc) / float(loss)
        elif acc is not None:
            logs['combine_metric'] = float(acc)
        elif loss is not None:
            logs['combine_metric'] = -float(loss)
        else:
            logs['combine_metric'] = 0.0
