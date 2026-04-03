# Dots OCR VLLM - Inference API

## CURL command

```shell
curl -X POST https://modal_generated_url \
     -F 'file=@/Users/mandamac1/Downloads/scie1.pdf' \
     -F 'prompt_mode=prompt_layout_all_en' \
     -F 'num_threads=32' \
     -F 'output_format=markdown_content'
```

Output formats are: ['json_content', 'markdown_content', 'markdown_nohf_content'] default set to json_content

**example** 

```shell
curl -X POST https://sana-khamaassi--dots-ocr-vllm-official-app-parse-documen-4bd76a.modal.run \
  -F 'file=@/Users/mandamac1/Downloads/scie1.pdf' \
  -F 'prompt_mode=prompt_layout_all_en' \
  -F 'num_threads=64' \
  -F 'output_format=markdown_content' \
  -o output.md
```