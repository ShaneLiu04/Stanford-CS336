#!/bin/sh
set -eu

HOSTNAME="modal-${MODAL_TASK_ID:-$(hostname)}"

tailscaled \
  --tun=userspace-networking \
  --state=mem: \
  --socks5-server=localhost:1080 \
  --outbound-http-proxy-listen=localhost:1080 &

tailscale up \
  --auth-key="${TAILSCALE_AUTHKEY}" \
  --hostname="${HOSTNAME}"

exec "$@"
