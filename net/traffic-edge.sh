#!/bin/sh
#
# Incoming traffic control with bandwidth, shared latency, and SFQ for specific IPs
#

DEV=eth0
RATE="200mbit"

tc qdisc del dev $DEV root

# Add root qdisc and set up HTB (Hierarchical Token Bucket)
tc qdisc add dev $DEV root handle 1: htb default 10

# Create root class for full bandwidth
tc class add dev $DEV parent 1: classid 1:1 htb rate ${RATE} burst 15k

# Create a subclass with bandwidth limit and apply netem delay
tc class add dev $DEV parent 1:1 classid 1:10 htb rate ${RATE} ceil ${RATE} burst 15k

tc qdisc add dev $DEV parent 1: handle 20: sfq perturb 10

# Show tc configuration
echo;echo "tc configuration for $DEV:"
tc qdisc show dev $DEV
tc class show dev $DEV
