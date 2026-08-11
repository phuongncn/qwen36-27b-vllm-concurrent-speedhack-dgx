ARG VLLM_BASE=vllm-qwen35-v3:latest
FROM ${VLLM_BASE}

COPY patches/patch_draft_vocab.py /opt/qwen36-speedhack/patch_draft_vocab.py
RUN python3 /opt/qwen36-speedhack/patch_draft_vocab.py \
 && grep -q DGX_SPARK_FR_SPEC_V1 \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/logits_processor.py \
 && grep -q DGX_SPARK_FR_SPEC_V1 \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5_mtp.py

COPY patches/patch_lmhead_x16.py /opt/qwen36-speedhack/patch_lmhead_x16.py
RUN python3 /opt/qwen36-speedhack/patch_lmhead_x16.py \
 && grep -q DGX_SPARK_INT8_LMHEAD_X16_V1 \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/logits_processor.py

ENTRYPOINT ["vllm"]
CMD ["serve", "--help"]
