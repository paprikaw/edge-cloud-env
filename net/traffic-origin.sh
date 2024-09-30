#!/bin/sh
#
# Incoming traffic control
#
DEV=eth0
RATE="400mbit"

tc qdisc del dev $DEV root
tc qdisc add dev $DEV root handle 1: htb default 10
tc class add dev $DEV parent 1: classid 1:1 htb rate ${RATE} burst 15k
tc class add dev $DEV parent 1:1 classid 1:10 htb rate ${RATE} ceil ${RATE} burst 15k
tc qdisc add dev $DEV parent 1:10 handle 10: sfq perturb 10

echo;echo "tc configuration for $DEV:"
tc qdisc show dev $DEV
tc class show dev $DEV