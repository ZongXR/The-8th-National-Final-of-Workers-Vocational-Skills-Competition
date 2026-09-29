# -*- coding: utf-8 -*-
import os
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, GemmaForSequenceClassification
from peft import PeftModel
from config import DATA_SET, MODEL_PATH

cur_dir = os.path.dirname(os.path.realpath(__file__))
lora_path = "%s/saved_model/%s" % (cur_dir, DATA_SET)


def val(label_file):
    test_data = load_dataset("json", data_files="%s/pre_data/%s/%s_dev.json" % (cur_dir, DATA_SET, DATA_SET), split="train")
    model = GemmaForSequenceClassification.from_pretrained(MODEL_PATH).to("cuda:0")
    model = PeftModel.from_pretrained(model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    pre_res = [0, 0]
    total = test_data.shape[0]
    print(f"total:{total}")
    preds = []
    for i, test_da in enumerate(test_data):
        if i % 1 == 0:
            print(f"当前第{i}例 ---> 进度{i * 100 / total:.2f}%")
        if DATA_SET in ("QQP", "MRPC", "RTE"):
            encoded_input = tokenizer(test_da["sentence1"], test_da["sentence2"], return_tensors='pt').to("cuda:0")
        else:
            encoded_input = tokenizer(test_da["sentence"], return_tensors='pt').to("cuda:0")
        with torch.no_grad():
            output = model(**encoded_input).logits
        pred = output.argmax().item()
        pre_res[pred] += 1
        preds.append(pred)

    with open(label_file, "w", encoding="utf8") as f:
        f.writelines("\n".join([str(x) for x in preds]))


if __name__ == '__main__':
    val("%s/result_%s.txt" % (cur_dir, DATA_SET))
