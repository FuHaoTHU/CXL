"""
Tokenizer utilities for the LLM model.
Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/tokenizer.py
"""
from typing import List, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"
def get_token_length_stats(text: str, tokenizer) -> dict:
    """获取文本的token统计信息"""
    tokens = tokenizer.encode(text)
    return {
        "length": len(tokens),
        "tokens": tokens
    }

def batch_process_texts(
    texts: List[str],
    tokenizer,
    process_func=None,
    batch_size: int = 1000,
    **kwargs
) -> List[str]:
    """批量处理文本，支持大规模数据集"""
    if process_func is None:
        process_func = generate_long_tail_tokens
        
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        processed = [
            process_func(text, tokenizer, **kwargs)
            for text in batch
        ]
        results.extend(processed)
        
        if (i // batch_size) % 10 == 0:  # 每10个batch打印一次进度
            print(f"Processed {i + len(batch)}/{len(texts)} texts")
            
    return results

if __name__ == '__main__':
    # 原有的测试代码
    tokenizer_13b = get_tokenizer(
        'facebook/opt-13b', trust_remote_code=True
    )
    tokenizer_66b = get_tokenizer(
        'facebook/opt-66b', trust_remote_code=True
    )
    
    # 添加分布测试
    sample_texts = ["Test text" * 10] * 100
    processed = batch_process_texts(
        sample_texts,
        tokenizer_13b,
        mean_length=2048,
        sigma=1.2,
        max_length=8192
    )
    
    # 验证处理后的分布
    lengths = [get_token_length_stats(text, tokenizer_13b)["length"] 
              for text in processed]
    validate_length_distribution(lengths)

def get_tokenizer(
        tokenizer_name: str,
        *args,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "llama" in tokenizer_name.lower() and kwargs.get("use_fast", True):
        print(
            "For some LLaMA-based models, initializing the fast tokenizer may "
            "take a long time. To eliminate the initialization time, consider "
            f"using '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer."
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, trust_remote_code=trust_remote_code, **kwargs
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA-based "
            f"model, use '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        print(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )
    return tokenizer


def detokenize_incrementally(
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        prev_output_tokens: List[str],
        new_token_id: int,
        skip_special_tokens: bool,
) -> Tuple[str, str]:
    """Detokenizes the new token in conjuction with the previous output tokens.

    NOTE: This function does not update prev_output_tokens.

    Returns:
        new_token: The new token as a string.
        output_text: The new output text as a string.
    """
    if skip_special_tokens and (new_token_id in tokenizer.all_special_ids):
        return None, prev_output_tokens
    new_token = tokenizer.convert_ids_to_tokens(
        new_token_id, skip_special_tokens=skip_special_tokens
    )
    output_tokens = prev_output_tokens + [new_token]

    # Convert the tokens to a string.
    # Optimization: If the tokenizer does not have `added_tokens_encoder`,
    # then we can directly use `convert_tokens_to_string`.
    if not getattr(tokenizer, "added_tokens_encoder", {}):
        output_text = tokenizer.convert_tokens_to_string(output_tokens)
        return new_token, output_text

    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE: The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    sub_texts = []
    current_sub_text = []
    for token in output_tokens:
        if skip_special_tokens and token in tokenizer.all_special_tokens:
            continue
        if token in tokenizer.added_tokens_encoder:
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    output_text = " ".join(sub_texts)
    return new_token, output_text


if __name__ == '__main__':
    tokenizer_13b = get_tokenizer(
        'facebook/opt-13b', trust_remote_code=True
    )
    tokenizer_66b = get_tokenizer(
        'facebook/opt-66b', trust_remote_code=True
    )
