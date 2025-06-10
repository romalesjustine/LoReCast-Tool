from transformers import EncoderDecoderModel

class SafeEncoderDecoderModel(EncoderDecoderModel):
    """
    Subclass of EncoderDecoderModel that removes any
    'num_items_in_batch' kwarg before forwarding to BART.
    """
    def forward(self, *args, **kwargs):
        # Strip out num_items_in_batch if it was passed by Seq2SeqTrainer
        if "num_items_in_batch" in kwargs:
            kwargs.pop("num_items_in_batch")
        return super().forward(*args, **kwargs)
