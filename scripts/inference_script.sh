
LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"

# # OmniASR inference
# OMNI_SCRIPT_PATH="/workspace/omniasr_inference.py"
# OMNI_SCRIPT_NAME=$(basename "$OMNI_SCRIPT_PATH" .py)
# OMNI_LOG_FILE="$LOG_DIR/${OMNI_SCRIPT_NAME}.log"
# /opt/conda/envs/omniasr/bin/python "$OMNI_SCRIPT_PATH" > "$OMNI_LOG_FILE" 2>&1

# Whisper inference
WHISPER_SCRIPT_PATH="/workspace/whisper_inference.py"
WHISPER_SCRIPT_NAME=$(basename "$WHISPER_SCRIPT_PATH" .py)
WHISPER_LOG_FILE="$LOG_DIR/${WHISPER_SCRIPT_NAME}.log"
/opt/conda/bin/python "$WHISPER_SCRIPT_PATH" > "$WHISPER_LOG_FILE" 2>&1
