
modprobe ifb numifbs=1
ip link set dev ifb0 up

# Remove any existing qdisc
tc qdisc del dev eth0 root
tc qdisc del dev ifb0 root

tc qdisc add dev eth0 handle ffff: ingress
tc filter add dev eth0 parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev ifb0