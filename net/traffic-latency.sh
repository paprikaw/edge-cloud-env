#!/bin/sh
#
# Incoming traffic control with bandwidth, shared latency, and SFQ for specific IPs
#

DEV=eth0
VM1_CIDR=172.26.133.218/32
VM2_CIDR=172.26.133.227/32
VM3_CIDR=172.26.132.14/32
VM4_CIDR=172.26.132.147/32
VM5_CIDR=172.26.133.107/32
LATENCY="50ms"


tc qdisc del dev $DEV root

# 添加根队列规则

otc qdisc add dev $DEV root handle 1: prio
tc qdisc add dev eth0 root handle 1: prio
tc qdisc add dev eth0 parent 1:1 handle 2: netem delay 50ms
sc add dev eth0 root handle 1: prio priomap 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2# tc qdisc add dev $DEV parent 10: handle 20: sfq perturb 10


# Apply filters for all VM CIDRs to use the same class and netem+sfq settings
tc filter add dev $DEV protocol ip parent 1: prio 1 u32 match ip dst $VM1_CIDR flowid 1:1
tc filter add dev $DEV protocol ip parent 1: prio 1 u32 match ip dst $VM2_CIDR flowid 1:1
tc filter add dev $DEV protocol ip parent 1: prio 1 u32 match ip dst $VM3_CIDR flowid 1:1
tc filter add dev $DEV protocol ip parent 1: prio 1 u32 match ip dst $VM4_CIDR flowid 1:1
tc filter add dev $DEV protocol ip parent 1: prio 1 u32 match ip dst $VM5_CIDR flowid 1:1


# Show tc configuration
echo;echo "tc configuration for $DEV:"
tc qdisc show dev $DEV
tc class show dev $DEV
