#!/bin/sh
#
# Incoming traffic control with bandwidth, shared latency, and SFQ for specific IPs
#

DEV=ifb0
RATE="2000mbit"
VM1_CIDR=115.146.93.94/32
VM2_CIDR=115.146.92.106/32
VM3_CIDR=115.146.94.60/32
VM4_CIDR=45.113.235.236/32
LATENCY="50ms"


tc qdisc del dev $DEV root

# Add root qdisc and set up HTB (Hierarchical Token Bucket)
tc qdisc add dev $DEV root handle 1: htb default 20

# Create root class for full bandwidth
tc class add dev $DEV parent 1: classid 1:1 htb rate ${RATE} burst 15k

# Create a subclass with bandwidth limit and apply netem delay
tc class add dev $DEV parent 1:1 classid 1:10 htb rate ${RATE} ceil ${RATE} burst 15k
# Default class for all other traffic
tc class add dev $DEV parent 1:1 classid 1:20 htb rate ${RATE} ceil ${RATE} burst 15k

# Add netem delay and then sfq for fairness
tc qdisc add dev $DEV parent 1:10 handle 10: netem delay $LATENCY

tc qdisc add dev $DEV parent 1: handle 20: sfq perturb 10

# Apply filters for all VM CIDRs to use the same class and netem+sfq settings
tc filter add dev $DEV protocol ip parent 1: prio 1 u32 match ip dst $VM1_CIDR flowid 1:10
tc filter add dev $DEV protocol ip parent 1: prio 1 u32 match ip dst $VM2_CIDR flowid 1:10
tc filter add dev $DEV protocol ip parent 1: prio 1 u32 match ip dst $VM3_CIDR flowid 1:10
tc filter add dev $DEV protocol ip parent 1: prio 1 u32 match ip dst $VM4_CIDR flowid 1:10

# Show tc configuration
echo;echo "tc configuration for $DEV:"
tc qdisc show dev $DEV
tc class show dev $DEV
