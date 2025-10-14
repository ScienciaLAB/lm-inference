curl -X POST "Modal url" \
  -H "accept: application/json" \
  -F "pdf=@/Users/mandamac1/Downloads/gre.pdf" \
  -F "dpi=150"


########## For vllm ###########@
curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "pdf=@/Users/mandamac1/Downloads/gre.pdf" \
  -F "dpi=150" \
  https://sana-khamassi5678--qwen-2-5-vl-7b-instruct-vllm-process--908916.modal.run