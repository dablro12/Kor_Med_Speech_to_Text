from transformers import WhisperTokenizer
import evaluate
import numpy as np
# 미리 metric 로더를 만들어서, 매번 load 성능 저하 방지
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred, tokenizer, metric_name='cer'):
    """
    metric_name: "wer" or "cer"
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # metric_name에 따라 wer 또는 cer 반환
    if metric_name.lower() == "wer":
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    else: # default is cer
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

if __name__ == "__main__":
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Korean", task="transcribe")

    pred = type("Pred", (), {})()
    pred.predictions = np.array([[1, 2, 3], [4, 5, 6]])
    pred.label_ids = np.array([[1, 2, 3], [1, 2, 3]])

    print(compute_metrics(pred, tokenizer, "wer"))
    print(compute_metrics(pred, tokenizer, "cer"))