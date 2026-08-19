#!/usr/bin/env bash
# Helper around the vast.ai CLI for this lab's target shape:
# 1x RTX 6000 Ada (48 GB) with enough NVMe for the FP8 checkpoint.
#
# Usage:
#   ./create-instance.sh search              # list candidate offers, cheapest first
#   ./create-instance.sh create <OFFER_ID>   # rent one, running the lab image
#   ./create-instance.sh status <INSTANCE_ID># show state + mapped port for 8000
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/foundry-ai-dev/vllm-fp8-lab:latest}"
DISK="${DISK:-120}"

case "${1:-}" in
  search)
    # High inet_down matters: the 28 GB weight download happens on the instance.
    vastai search offers \
      'gpu_name=RTX_6000Ada num_gpus=1 disk_space>=120 inet_down>=500 rentable=true verified=true' \
      -o 'dph'
    ;;
  create)
    OFFER_ID="${2:?usage: $0 create <OFFER_ID>}"
    # ssh launch mode + onstart: vast.ai's ssh mode does not run the image
    # ENTRYPOINT, so onstart backgrounds it and logs to /workspace/vllm.log.
    vastai create instance "$OFFER_ID" \
      --image "$IMAGE" \
      --disk "$DISK" \
      --env '-p 8000:8000' \
      --ssh --direct \
      --onstart-cmd 'nohup /usr/local/bin/entrypoint.sh > /workspace/vllm.log 2>&1 &'
    echo
    echo "Watch progress:   $0 status <INSTANCE_ID>"
    echo "Tail server log:  ssh into the instance, then: tail -f /workspace/vllm.log"
    ;;
  status)
    INSTANCE_ID="${2:?usage: $0 status <INSTANCE_ID>}"
    vastai show instance "$INSTANCE_ID"
    echo
    echo "The external port mapped to 8000 appears in the ports column above;"
    echo "the API base URL is http://<public_ip>:<mapped_port>/v1"
    ;;
  *)
    grep '^# ' "$0" | sed 's/^# //'
    exit 1
    ;;
esac
