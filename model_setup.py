from transformers import LongformerTokenizerFast
from model_wrapper import SafeEncoderDecoderModel

def get_model():
    """
    Returns a SafeEncoderDecoderModel (LongFormer encoder + BART decoder)
    plus its tokenizer. Any num_items_in_batch kwarg will be removed by the wrapper.
    """
    tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")

    model = SafeEncoderDecoderModel.from_encoder_decoder_pretrained(
        "allenai/longformer-base-4096",
        "facebook/bart-base"
    )

    # Configure decoder tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.max_length = 200
    model.config.no_repeat_ngram_size = 3

    return model, tokenizer
